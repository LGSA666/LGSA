import random
import json
from transformers import AutoTokenizer
import torch
from torch.utils.data import Subset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import os
import datasets
from tqdm import tqdm
import argparse
import swanlab
import yaml
import time 

from eval import evaluate, evaluate_based_on_path

import utils

import numpy as np

from models.loss import HierarchicalGofLoss
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
import torch.nn.functional as F


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data', type=str, default='rcv1')
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--update', type=int, default=1)
    parser.add_argument('--model', type=str, default='prompt')
    parser.add_argument('--arch', type=str, default='/your model path/')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--graph', type=str, default='GAT')
    parser.add_argument('--low-res', default=False, action='store_true')
    parser.add_argument('--seed', default=3, type=int)

    parser.add_argument('--gof_loss_weight', type=float, default=1.0, help="Weight for the Hierarchical GOF regularization loss.") # 0.01
    parser.add_argument('--gof_angle', type=float, default=10.0, help="Target angle in degrees for parent-child similarity.") # 15.0
    parser.add_argument('--gof_decay', type=float, default=0.90, help="Target length decay factor for parent-child norms.") # 0.9
    parser.add_argument('--align_w', type=float, default=0.05, help="Weight for the ideal embedding alignment loss.")
    parser.add_argument('--swanlab', default=False, action='store_true')
    

    return parser


class Save:
    def __init__(self, model, optimizer, scheduler, args):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def __call__(self, score, best_score, name):
        torch.save({'param': self.model.state_dict(),
                    'optim': self.optimizer.state_dict(),
                    'sche': self.scheduler.state_dict() if self.scheduler is not None else None,
                    'score': score, 'args': self.args,
                    'best_score': best_score},
                   name)

def get_exponential_with_warmup_scheduler(optimizer, warmup_steps, total_steps, gamma):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            decay_steps = (current_step - warmup_steps) // warmup_steps
            return gamma ** decay_steps
    return LambdaLR(optimizer, lr_lambda)

if __name__ == '__main__':
    parser = parse()
    args = parser.parse_args()
    print(args)
    utils.seed_torch(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.arch)
    data_path = os.path.join('data', args.data)
    args.name = args.data + '-' + args.name
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))

    if args.swanlab:
        swanlab.init(config=args, project='LGSA-GOF')
    logger = utils.init_logger(os.path.join('checkpoints', args.name, 'run.log'))
    logger.info(args)
    batch_size = args.batch

    label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
    label_dict = {i: v for i, v in label_dict.items()}
    num_class = len(label_dict)
    slot2value = torch.load(os.path.join(data_path, 'slot.pt'))
    value2slot = {}
    # num_class = 0
    for s in slot2value:
        if s >= num_class:
            continue
        for v in slot2value[s]:
            if v < num_class:
                value2slot[v] = s

    # num_class += 1
    path_list = [(i, v) for v, i in value2slot.items()]
    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1


    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth


    depth_dict = {i: get_depth(i) for i in range(num_class)} 
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))
    
    logger.info('num_class: {}'.format(num_class))
    logger.info('label dict: {}'.format(label_dict))
    logger.info('slot2value: {}'.format(slot2value))
    logger.info('value2slot: {}'.format(value2slot))
    logger.info('depth2label: {}'.format(depth2label))
    logger.info('path_list: {}'.format(path_list))

    if args.model == 'prompt':
        if os.path.exists(os.path.join(data_path, args.model)):
            dataset = datasets.load_from_disk(os.path.join(data_path, args.model))
        else:
            dataset = datasets.load_dataset('json',
                                            data_files={'train': 'data/{}/{}_train.json'.format(args.data, args.data),
                                                        'dev': 'data/{}/{}_dev.json'.format(args.data, args.data),
                                                        'test': 'data/{}/{}_test.json'.format(args.data, args.data), })

            prefix = [] 
            for i in range(max_depth):
                prefix.append(tokenizer.vocab_size + num_class + i)
                prefix.append(tokenizer.vocab_size + num_class + max_depth)
            prefix.append(tokenizer.sep_token_id)


            def data_map_function(batch, tokenizer): 
                new_batch = {'input_ids': [], 'token_type_ids': [], 'attention_mask': [], 'labels': []}
                for l, t in zip(batch['label'], batch['token']):
                    new_batch['labels'].append([[-100 for _ in range(num_class)] for _ in range(max_depth)]) 
                    for d in range(max_depth): 
                        for i in depth2label[d]: 
                            new_batch['labels'][-1][d][i] = 0
                        for i in l:
                            if new_batch['labels'][-1][d][i] == 0:
                                new_batch['labels'][-1][d][i] = 1
                    new_batch['labels'][-1] = [x for y in new_batch['labels'][-1] for x in y] 

                    tokens = tokenizer(t, truncation=True)
                    new_batch['input_ids'].append(tokens['input_ids'][:-1][:512 - len(prefix)] + prefix) 
                    new_batch['input_ids'][-1].extend(
                        [tokenizer.pad_token_id] * (512 - len(new_batch['input_ids'][-1]))) 
                    new_batch['attention_mask'].append(
                        tokens['attention_mask'][:-1][:512 - len(prefix)] + [1] * len(prefix))
                    new_batch['attention_mask'][-1].extend([0] * (512 - len(new_batch['attention_mask'][-1])))
                    new_batch['token_type_ids'].append([0] * 512)

                return new_batch


            dataset = dataset.map(lambda x: data_map_function(x, tokenizer), batched=True)
            dataset.save_to_disk(os.path.join(data_path, args.model))
        dataset['train'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['dev'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])
        dataset['test'].set_format('torch', columns=['attention_mask', 'input_ids', 'labels'])

        logger.info("train_data num is: {}".format(len(dataset['train'])))
        logger.info("dev_data num is: {}".format(len(dataset['dev'])))
        logger.info("test_data num is: {}".format(len(dataset['test'])))

        from models.prompt import Prompt

    else:
        raise NotImplementedError
    if args.low_res:
        if os.path.exists(os.path.join(data_path, 'low.json')):
            index = json.load(open(os.path.join(data_path, 'low.json'), 'r'))
        else:
            index = [i for i in range(len(dataset['train']))]
            random.shuffle(index)
            json.dump(index, open(os.path.join(data_path, 'low.json'), 'w'))
        dataset['train'] = dataset['train'].select(index[len(index) // 5:len(index) // 10 * 3])
    model = Prompt.from_pretrained(args.arch, num_labels=len(label_dict), path_list=path_list, layer=args.layer,
                                   graph_type=args.graph, data_path=data_path, depth2label=depth2label) 
    model.init_embedding()
    logger.info(model)
    logger.info(f"Total params: {sum(param.numel() for param in model.parameters()) / 1000000.0}M. ")

    model.to('cuda')

    gof_loss_fn = HierarchicalGofLoss(
        depth2label=depth2label,
        value2slot=value2slot,
        parent_child_angle_deg=args.gof_angle,
        length_decay_factor=args.gof_decay,
        return_components=True
    ).to('cuda')

    train = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, )
    dev = DataLoader(dataset['dev'], batch_size=8, shuffle=False)
    model.to('cuda')
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_initial = args.lr
    lr_final = 1e-7
    gamma49 = (lr_final / lr_initial) ** (1 / 49)
    if args.data == 'WebOfScience':
        warmup_steps = (30070 // args.batch) + 1
        total_steps = warmup_steps * 50
    elif args.data == 'rcv1':
        warmup_steps = (20834 // args.batch) + 1
        total_steps = warmup_steps * 50
    else: # nyt
        warmup_steps = (23391 // args.batch) + 1
        total_steps = warmup_steps * 50
    scheduler = get_exponential_with_warmup_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, gamma=gamma49)



    save = Save(model, optimizer, None, args)
    best_score_macro = 0
    best_score_micro = 0
    update_step = 0
    loss = 0
    loss_total = 0
    if not os.path.exists(os.path.join('checkpoints', args.name)):
        os.mkdir(os.path.join('checkpoints', args.name))
    
    def numParams(net):
        num = 0
        for param in net.parameters():
            if param.requires_grad:
                num += int(np.prod(param.size()))
        return num
    print("numParams:", numParams(model))

    for epoch in range(args.epoch):
        logger.info("------------ epoch {} ------------".format(epoch + 1))
        start_time = time.time()
        print("start_time:", start_time)
        model.train()
        with tqdm(train) as p_bar:
            for batch in p_bar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output, masked_lm_loss, multiclass_loss = model(**batch) 

                graph_embedding_layer = model.get_input_embeddings()  
                label_weight_function = graph_embedding_layer.weight()[-num_class - len(depth2label) - 1: -len(depth2label)-1]

                geometric_loss, geo_comps = gof_loss_fn(label_weight_function)

                # ideal = model.ideal_label_embeddings.to(label_weight_function.device)
                # cur = F.normalize(label_weight_function, dim=1)
                # idt = F.normalize(ideal, dim=1)
                # align_loss = (1.0 - (cur * idt).sum(dim=1)).mean()
                # swanlab.log({"loss/align": align_loss.item()})

                # align_w = 0.05
                # loss_total = output['loss'] + args.gof_loss_weight * geometric_loss + align_w * align_loss

                warmup_ratio = 0.1
                global_step = epoch * len(train) + p_bar.n
                total_steps = scheduler.total_steps if hasattr(scheduler, "total_steps") else (args.epoch * len(train))
                alpha = min(1.0, global_step / max(1, int(warmup_ratio * total_steps)))

                loss_total = output['loss'] + (args.gof_loss_weight * alpha) * geometric_loss #  + args.align_w * align_loss

                loss_total.backward()
                loss += loss_total.item()
                update_step += 1
                if update_step % args.update == 0:
                    if args.swanlab:
                        swanlab.log({'loss': loss, 'masked_lm_loss': masked_lm_loss.item(), 'multiclass_loss': multiclass_loss.item(), 'geometric_loss': geometric_loss.item(), 'align_loss': align_loss.item()})
                    p_bar.set_description(
                        'loss:{:.4f}'.format(loss, ))
                    optimizer.step()
                    scheduler.step()  
                    optimizer.zero_grad()
                    loss_total = 0
                    loss = 0
                    update_step = 0

        end_time = time.time()
        print("end_time:", end_time)
        elapsed_train_time = end_time - start_time
        print(f"epoch train time:{elapsed_train_time}s")

        model.eval()
        pred = []
        gold = []
        with torch.no_grad(), tqdm(dev) as pbar:
            start_time = time.time()
            print("start_time:", start_time)
            for batch in pbar:
                batch = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, )
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)
            end_time = time.time()
            print("end_time:", end_time)
            elapsed_eval_time = end_time - start_time
            print(f"epoch eval time:{elapsed_eval_time}s")

        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        logger.info(' macro: {:.4f}, micro: {:.4f}'.format(macro_f1, micro_f1))
        print('macro', macro_f1, 'micro', micro_f1)

        if args.swanlab:
            swanlab.log({'val_macro': macro_f1, 'val_micro': micro_f1})


        if macro_f1 > best_score_macro:
            best_score_macro = macro_f1
            save(macro_f1, best_score_macro, os.path.join('checkpoints', args.name, 'checkpoint_best_macro.pt'))
            logger.info(f"New best macro F1: {best_score_macro:.4f}. Checkpoint saved.")
            # early_stop_count = 0

        if micro_f1 > best_score_micro:
            best_score_micro = micro_f1
            save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_best_micro.pt'))
            logger.info(f"New best micro F1: {best_score_micro:.4f}. Checkpoint saved.")

        save(micro_f1, best_score_micro, os.path.join('checkpoints', args.name, 'checkpoint_last.pt'))
        if args.swanlab:
            swanlab.log({'best_macro': best_score_macro, 'best_micro': best_score_micro})

        torch.cuda.empty_cache()

    # test
    test = DataLoader(dataset['test'], batch_size=16, shuffle=False)
    model.eval()


    def test_function(extra):
        checkpoint = torch.load(os.path.join('checkpoints', args.name, 'checkpoint_best{}.pt'.format(extra)),
                                map_location='cpu')
        logger.info(f'Test load checkpoint: {checkpoint}')
        model.load_state_dict(checkpoint['param'])
        pred = []
        gold = []
        with torch.no_grad(), tqdm(test) as pbar:
            for batch in pbar:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                output_ids, logits = model.generate(batch['input_ids'], depth2label=depth2label, )
                for out, g in zip(output_ids, batch['labels']):
                    pred.append(set([i for i in out]))
                    gold.append([])
                    g = g.view(-1, num_class)
                    for ll in g:
                        for i, l in enumerate(ll):
                            if l == 1:
                                gold[-1].append(i)

        scores = evaluate(pred, gold, label_dict)
        macro_f1 = scores['macro_f1']
        micro_f1 = scores['micro_f1']
        precision = scores['precision']
        recall = scores['recall']
        logger.info('macro: {}, micro: {}, precision: {}, recall: {}'.format(macro_f1, micro_f1, precision, recall))
        print('macro', macro_f1, 'micro', micro_f1)
        print("---------------------")
        print(scores['full'])

        scores = evaluate_based_on_path(pred, gold, label_dict, value2slot, slot2value)
        c_micro_f1 = scores['c_micro_f1']
        c_macro_f1 = scores['c_macro_f1']
        logger.info('c_micro_f1: {}, c_macro_f1: {}'.format(c_micro_f1, c_macro_f1))

        with open(os.path.join('checkpoints', args.name, 'result{}.txt'.format(extra)), 'w') as f:
            print('macro', macro_f1, 'micro', micro_f1, file=f)
            prefix = 'test' + extra
        if args.swanlab:
            swanlab.log({prefix + '_macro': macro_f1, prefix + '_micro': micro_f1})


    test_function('_macro')
    test_function('_micro')
