import torch
import math
import tqdm
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model

class CG(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, is_group_in_train, group_info):
        # initialize model
        model = initialize_model(config, d_out).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.logged_fields.append('group_alpha')
        
        self.outer_lr = config.lr

        self.device = config.device
        self.C = config.cg_C
        _, self.g_counts = group_info
        self.g_counts = torch.tensor(self.g_counts, device=self.device, dtype=torch.float32)
        wts, self.adj_wts = torch.zeros_like(self.g_counts), torch.zeros_like(self.g_counts)
        wts[self.g_counts>0] = torch.exp(self.C/torch.sqrt(self.g_counts[self.g_counts>0]))
        self.wts = wts/wts.sum()
        self.adj_wts[self.g_counts>0] = self.C/torch.sqrt(self.g_counts[self.g_counts>0])
        print ("Using up-weight: ", self.wts.cpu().numpy())
        print ("Using adj-weight: ", self.adj_wts.cpu().numpy())
        
        self.config = config
        self.batch_size = config.batch_size
        self.num_groups = grouper.n_groups
        self.alpha = torch.autograd.Variable(torch.ones(self.num_groups, device=self.device)*(1./self.num_groups), requires_grad=True)
        self.lmbda = torch.autograd.Variable(torch.zeros(self.num_groups, device=self.device), requires_grad=True)
        self.rwt = torch.autograd.Variable(self.wts.clone(), requires_grad=False)
        self.step_size = config.cg_step_size

    def process_batch(self, batch):
        """
        A helper function for update() and evaluate() that processes the batch
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
              all Tensors are of size (batch_size,)
        """
        results = super().process_batch(batch)
        results['group_alpha'] = self.alpha.detach()
        return results

    def objective(self, results):
        # compute group losses
        group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'][:self.batch_size],
            results['y_true'][:self.batch_size],
            results['g'][:self.batch_size],
            self.num_groups,
            return_dict=False)
                
        loss = group_losses @ self.rwt
        return loss

    def _params(self):
        if self.config.model.find('bert') >= 0:
            params = []
            select = ['layer.10', 'layer.11', 'bert.pooler.dense', 'classifier']
            for name, param in self.model.named_parameters():
                for s in select:
                    if (name.find(s) >= 0):
                        params.append(param)
                        break
            return params
        elif self.config.model.startswith('resnet50'):
            params = []
            select = ['layer3', 'layer4', 'fc.weight', 'fc.bias']
            for name, param in self.model.named_parameters():
                for s in select:
                    if (name.find(s) >= 0):
                        params.append(param)
                        break
            return params
        elif self.config.model.startswith('densenet121'):
            params = []
            select = ['features.denseblock4', 'classifier']
            for name, param in self.model.named_parameters():
                for s in select:
                    if (name.find(s) >= 0):
                        params.append(param)
                        break
            return params
        else:
            return list(self.model.parameters())  
    
    def _update(self, results):
        """
        Process the batch, update the log, and update the model, group weights, and scheduler.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch, such as:
                - g (Tensor)
                - y_true (Tensor)
                - metadata (Tensor)
                - loss (Tensor)
                - metrics (Tensor)
                - objective (float)
        """      
        group_losses, _, _ = self.loss.compute_group_wise(
            results['y_pred'],
            results['y_true'],
            results['g'],
            self.num_groups,
            return_dict=False)        
        params = self._params()
        all_grads = [None]*self.num_groups
        for li in range(self.num_groups):
            all_grads[li] = torch.autograd.grad(group_losses[li], params, retain_graph=True)
            assert all_grads[li] is not None
        
        RTG = torch.zeros([self.num_groups, self.num_groups], device=self.device)
        for li in range(self.num_groups):
            for lj in range(self.num_groups):
                dp = 0
                vec1_sqnorm, vec2_sqnorm = 0, 0
                for pi in range(len(params)):
                    fvec1 = all_grads[lj][pi].detach().flatten()
                    fvec2 = all_grads[li][pi].detach().flatten()
                    dp += fvec1 @ fvec2
                    vec1_sqnorm += torch.norm(fvec1)**2
                    vec2_sqnorm += torch.norm(fvec2)**2
                RTG[li, lj] = dp/torch.clamp(torch.sqrt(vec1_sqnorm*vec2_sqnorm), min=1e-3)

        _gl = torch.sqrt(group_losses.detach().unsqueeze(-1))
        RTG = torch.mm(_gl, _gl.t()) * RTG
        _exp = self.step_size*(RTG @ self.wts)
        
        # to avoid overflow
        _exp -= _exp.max()
        self.alpha.data = torch.exp(_exp)
        self.rwt *= self.alpha.data
        self.rwt = self.rwt/self.rwt.sum()
        self.rwt = torch.clamp(self.rwt, min=1e-5)
            
        results['group_alpha'] = self.rwt.detach()
        # update model
        super()._update(results)
