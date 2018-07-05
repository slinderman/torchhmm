import numpy as np
import torch
from torch.autograd import Function, Variable
import torchhmm.hmm as hmm

# TODO: cython code expects doubles.  We've been working with floats.

# Pytorch extension
class HMMNormalizer(Function):
    @staticmethod
    def forward(ctx, log_pi0, log_As, log_likes):
        T, K = log_likes.shape
        alphas = np.zeros((T, K), dtype=np.float32)
        to_numpy = lambda arr: arr.data.numpy() if not arr.is_cuda else arr.cpu().data.numpy()
        
        Z = hmm.forward_pass(to_numpy(log_pi0), to_numpy(log_As), to_numpy(log_likes), alphas)
        ctx.save_for_backward(log_As)
        ctx.alphas = alphas
        return torch.FloatTensor([Z])

    @staticmethod
    def backward(ctx, grad_output):
        to_numpy = lambda arr: arr.data.numpy() if not arr.is_cuda else arr.cpu().data.numpy()
        log_As = to_numpy(ctx.saved_tensors[0])
        alphas = ctx.alphas
        T, K = alphas.shape

        d_log_pi0 = np.zeros(K,dtype=np.float32)
        d_log_As = np.zeros((T - 1, K, K),dtype=np.float32)
        d_log_likes = np.zeros((T, K),dtype=np.float32)

        hmm.backward_pass(log_As, alphas, d_log_pi0, d_log_As, d_log_likes)

        return Variable(torch.FloatTensor(d_log_pi0)) * grad_output, \
               Variable(torch.FloatTensor(d_log_As)) * grad_output, \
               Variable(torch.FloatTensor(d_log_likes)) * grad_output

hmm_marginal_likelihood = HMMNormalizer.apply
