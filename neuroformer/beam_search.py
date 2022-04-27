import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue

device = 'cpu'

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50     # should be block_size



class BeamSearchNode(object):
    def __init__(self, decoder_input, previousNode, Id, logProb, length):
        '''
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.decoder_input = decoder_input
        self.prevNode = previousNode
        self.id = Id
        self.logp = logProb
        self.leng = length
    
    def eval(self, alpha=1.0):
        reward = 0
        # add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

@torch.no_grad()
def beam_decode(decoder, stoi, x, frame_end=0):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] 
    :where B is the batch_size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H]
    '''
    tf = frame_end
    beam_width = 5
    topk = 1    # how many sentences do you want to generate
    decoded_batch = []

    t = x['id'].shape[-1]
    pad = x['pad']
    x['id_full'] = x['id'][:, 0]
    x['id'] = x['id'][:, 0]
    # decoding goes sentence by sentence
    for idx in range(x['id'].shape[0]):
        t_pad = torch.tensor([stoi['PAD']] * (t - x['id_full'].shape[-1]))
        x['id'] = torch.cat((x['id_full'], t_pad)).unsqueeze(0)

        # start with the start of the sentence token
        input_id_token = x['id'][:, 0]
        decoder_input = x

        # number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node = dec_output vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_input, None, input_id_token, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            try:
                score, n = nodes.get()
            except: TypeError
            decoder_input = n.decoder_input

            # if n.leng >= (t - pad + 1):  # and n.prevNode != None:
            if n.id.item() == stoi['EOS'] or n.leng >= x['id'].shape[1] and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum number of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue
            
            # decode for one step using decoder
            # x['id'] = torch.where(x['id'] <= stoi['PAD'], x['id'], stoi['PAD'])
            preds, _, _ = decoder(x)
            # use logits of n.leng - 1 decoding step
            decoder_output = torch.log(F.softmax(preds['id'][:, tf + (n.leng - 1)], dim=-1))

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            # indexes = torch.where(indexes <= stoi['PAD'], indexes, stoi['PAD'])
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                
                # replace padding token with prediction 
                decoder_input = n.decoder_input
                decoder_input['id'][:, n.leng] = decoded_t
                
                node = BeamSearchNode(decoder_input, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # puth them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                try:
                    nodes.put((score, nn))
                except TypeError:
                    continue
                # increase qsize
            qsize += len(nextnodes) - 1
        
        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = []
            for _ in range(topk):
                try:
                    endnodes.append(nodes.get())
                except TypeError:
                    continue
            # endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.id)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.id)

            utterance = utterance[::-1][1:]
            utterances.append(utterance)

        decoded_batch.append(utterances)
        
    return torch.tensor(decoded_batch).flatten()
