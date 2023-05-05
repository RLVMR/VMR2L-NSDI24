import torch
import torch.nn as nn


class QuantileLossCalculator:
    def __init__(self, quantiles, missing_value, device):
        self.quantiles = torch.FloatTensor(quantiles).to(device)
        assert ((self.quantiles > 0) & (
                self.quantiles < 1)).all(), f'Quantiles must be between 0 and 1, but we have {quantiles}...'
        self.missing_value = missing_value

    def loss_fn(self, parameters: torch.Tensor, trans, ids, time, labels: torch.Tensor, weights: torch.Tensor):
        """
        Compute using quantile loss which needs to be minimized.
        Ignore time steps where labels are missing.
        Args:
            parameters ([batch_size, time_len, 2]): estimated distribution parameters (mu, sigma);
            v_batch ([batch_size, 2, 1]): scaling factor for the current batch;
            labels ([batch_size, time_len]): ground truth z_t;
            weights ([batch_size, time_len]): weighing each time step differently.
        Returns:
            loss (Variable): average log-likelihood loss across the batch
        """
        parameters = trans.scale_back(parameters, instance_id=ids, time_col=time)
        zero_index = (labels != self.missing_value)
        nonmissing_weights = weights[zero_index]

        loss = 0.
        for i, quantile in enumerate(self.quantiles):
            diff = (labels - parameters[:, :, i])[zero_index] * nonmissing_weights
            loss += quantile * torch.sum(torch.clamp(diff, min=0.)) + \
                    (1. - quantile) * torch.sum(torch.clamp(-diff, min=0.))
        return loss


class MeanLossCalculator:
    def __init__(self, missing_value):
        self.missing_value = missing_value
        self.l1loss = nn.L1Loss()

    def loss_fn(self, parameters: torch.Tensor, trans, ids, time, labels: torch.Tensor, weights: torch.Tensor):
        """
        Compute using gaussian the log-likelihood which needs to be maximized. Ignore time steps where labels are missing.
        Args:
            parameters ([batch_size, time_len, 2]): estimated distribution parameters (mu, sigma);
            v_batch ([batch_size, 2, 1]): scaling factor for the current batch;
            labels ([batch_size, time_len]): ground truth z_t;
            weights ([batch_size, time_len]): weighing each time step differently.
        Returns:
            loss (Variable): average log-likelihood loss across the batch
        """
        parameters = trans.scale_back(parameters, instance_id=ids, time_col=time)
        zero_index = (labels != self.missing_value)
        l1 = torch.abs(parameters[:, :, 0][zero_index] - labels[zero_index])
        loss = torch.mean(torch.mul(l1, weights[zero_index]))
        return loss


class Beta:
    def __init__(self, missing_value):
        self.softplus = nn.Softplus()
        self.missing_value = missing_value

    def loss_fn(self, parameters: torch.Tensor, trans, ids, time, labels: torch.Tensor, weights: torch.Tensor):
        """
        Compute using beta distribution the log-likelihood which needs to be maximized. Ignore time steps where labels are missing.
        Args:
            parameters ([batch_size, time_len, 2]): estimated distribution parameters (alpha, beta);
            v_batch ([batch_size, 2, 1]): scaling factor for the current batch;
            labels ([batch_size, time_len]): ground truth z_t;
            missing_value (float): a single number representing missing values, which will not count towards loss;
            use_boundary (boolean): not applicable for beta distributions.
        Returns:
            loss (Variable): average log-likelihood loss across the batch
        """
        SMALL_CONST = 1e-4
        labels = torch.clamp(labels, min=SMALL_CONST, max=1 - SMALL_CONST)  # to avoid nan.
        parameters = self.softplus(parameters)
        alpha = parameters[:, :, 0]
        beta = parameters[:, :, 1]
        zero_index = (labels != self.missing_value)
        distribution = torch.distributions.beta.Beta(alpha[zero_index], beta[zero_index])
        return -torch.mean(torch.mul(distribution.log_prob(labels[zero_index]), weights[zero_index]))

    def emit_mean(self, parameters: torch.Tensor):
        parameters = self.softplus(parameters)
        alpha = parameters[:, :, 0]
        beta = parameters[:, :, 1]
        mean = torch.div(alpha, alpha + beta)
        return mean

    def emit_mean_one_step(self, parameters: torch.Tensor):
        parameters = self.softplus(parameters)
        alpha = parameters[:, 0]
        beta = parameters[:, 1]
        mean = torch.div(alpha, alpha + beta)
        return mean

    def emit_sample(self, parameters: torch.Tensor):
        parameters = self.softplus(parameters)
        alpha = parameters[:, :, 0]
        beta = parameters[:, :, 1]
        return torch.distributions.beta.Beta(alpha, beta).sample()

    def emit_sample_one_step(self, parameters: torch.Tensor):
        parameters = self.softplus(parameters)
        alpha = parameters[:, 0]
        beta = parameters[:, 1]
        return torch.distributions.beta.Beta(alpha, beta).sample()


class Categorical:
    def __init__(self, num_class):
        self.param_size = num_class
        self.emit = nn.Softmax(dim=-1)

    def loss_fn(self, parameters: torch.Tensor, trans, ids, time, labels: torch.Tensor, weights: torch.Tensor):
        return F.cross_entropy(parameters.view(-1, self.param_size), labels.view(-1, self.param_size), weight=weights)

    def emit_mean(self, parameters: torch.Tensor):
        return torch.argmax(parameters, dim=-1)

    def emit_mean_one_step(self, parameters: torch.Tensor):
        return torch.argmax(parameters, dim=-1)

    def emit_sample(self, parameters: torch.Tensor):
        dist = torch.distributions.categorical.Categorical(logits=parameters)
        return dist.sample()

    def emit_sample_one_step(self, parameters: torch.Tensor):
        dist = torch.distributions.categorical.Categorical(logits=parameters)
        return dist.sample()


class Gaussian:
    def __init__(self, missing_value):
        self.param_size = 2
        self.missing_value = missing_value
        self.softplus = nn.Softplus()

    def loss_fn(self, parameters: torch.Tensor, trans, ids, time, labels: torch.Tensor, weights: torch.Tensor):
        """
        Compute using gaussian the log-likelihood which needs to be maximized. Ignore time steps where labels are missing.
        Args:
            parameters ([batch_size, 2]): estimated distribution parameters at time step t (mu, sigma);
            v_batch ([batch_size, 2]): scaling factor for the current batch;
            labels ([batch_size]): ground truth z_t for the current time step;
            weights ([batch_size, time_len]): weighing each time step differently.
        Returns:
            loss (Variable): average log-likelihood loss across the batch
        """
        sigma = self.softplus(parameters[:, :, 1])
        parameters = trans.scale_back(torch.stack([parameters[:, :, 0], sigma], dim=-1),
                                      instance_id=ids, time_col=time)
        zero_index = (labels != self.missing_value)
        distribution = torch.distributions.normal.Normal(parameters[:, :, 0][zero_index],
                                                         parameters[:, :, 1][zero_index])
        return -torch.mean(torch.mul(distribution.log_prob(labels[zero_index]), weights[zero_index]))

    def emit_mean(self, parameters: torch.Tensor):
        return parameters[:, :, 0]

    def emit_mean_one_step(self, parameters: torch.Tensor):
        return parameters[:, 0]

    def emit_sample(self, parameters: torch.Tensor):
        distribution = torch.distributions.normal.Normal(parameters[:, :, 0], self.softplus(parameters[:, :, 1]))
        return distribution.sample()

    def emit_sample_one_step(self, parameters: torch.Tensor):
        distribution = torch.distributions.normal.Normal(parameters[:, 0], self.softplus(parameters[:, 1]))
        return distribution.sample()


name_to_dist = {
    'Gaussian': Gaussian,
    'Beta': Beta
}
