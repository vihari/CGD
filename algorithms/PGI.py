import torch
from models.initializer import initialize_model
from algorithms.single_model_algorithm import SingleModelAlgorithm
from wilds.common.utils import split_into_groups
from configs import tpu_utils

class PGI(SingleModelAlgorithm):
    """
    Predictive Group Invariance
    """
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, group_info):
        # initialize models
        featurizer = initialize_model(config, d_out=None).to(config.device)
        classifier = torch.nn.Linear(featurizer.d_out, d_out).to(config.device)
        model = torch.nn.Sequential(featurizer, classifier).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # algorithm hyperparameters
        self.penalty_weight = config.pgi_penalty_weight
        # additional logging
#         self.logged_fields.append('penalty')
        # set model components
        self.featurizer = featurizer
        self.classifier = classifier
        self.group_counts, _ = group_info
        print ("Group counts:", self.group_counts)

    def pgi_penalty(self, mean_x, mean_y):
        eps = 1e-3
        mean_x, mean_y = torch.clamp(mean_x, eps, 1-eps), torch.clamp(mean_y, eps, 1-eps)
        kl_dist = torch.sum(mean_x*torch.log(mean_x/mean_y))
        
        return kl_dist

    def process_batch(self, batch):
        """
        Override
        """
        # forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        features = self.featurizer(x)
        outputs = self.classifier(features)
        tpu_utils.mark_step()
        
        # package the results
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
            'features': outputs,
            }
        return results

    def objective(self, results):
        # extract features
        features = results.pop('features')

        if self.is_training:
            # split into groups
            unique_groups, group_indices, _ = split_into_groups(results['g'])
            
            """
            TODO:
            The following piece of code assumes there are only two groups. 
            Should be fine for most of the sub-population shift datasets: CMNIST, WaterBirds, CelebA, MultiNLI
            """
            # compute penalty
            n_groups_per_batch = unique_groups.numel()
            penalty = torch.zeros(1, device=self.device)
            n_labels = features.shape[-1]
            # assert n_labels == n_labels_in_batch, "Found %d %d %s" % (n_labels, n_labels_in_batch, unique_groups)
            # n_labels x batch_size -- so as to scatter add for each label
            g = torch.stack([results['g']]*n_labels, dim=1)
            # compute mean label probs per group
            agg_probs = torch.zeros([n_labels*2, n_labels], device=self.device).scatter_add_(dim=0, index=g, src=features)
            agg_num = torch.zeros([n_labels*2, n_labels], device=self.device).scatter_add_(dim=0, index=g, src=torch.ones_like(features))
            agg_probs = agg_probs/torch.clamp(agg_num, 1)
            unique_groups_lst = list(unique_groups.detach().cpu().numpy())
            for li in range(n_labels):
                a, b = 2*li, 2*li+1
                if (agg_probs[a].sum()==0) or (agg_probs[b].sum()==0):
                    continue
                # keep the easy group (larger one) second 
                if self.group_counts[a] > self.group_counts[b]:
                    a, b = b, a
                penalty += self.pgi_penalty(agg_probs[a], agg_probs[b])
            tpu_utils.mark_step()
        else:
            penalty = 0.

#         if isinstance(penalty, torch.Tensor):
#             results['penalty'] = penalty.item()
#         else:
#             results['penalty'] = penalty


        avg_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)

        return avg_loss + penalty * self.penalty_weight
