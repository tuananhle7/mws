import torch


class ClusterDist(torch.distributions.Distribution):
    def __init__(self, num_data, batch_shape=torch.Size([])):
        super().__init__()
        self.num_data = num_data
        self._batch_shape = batch_shape

    def next_dist(self, z):
        """
        Args:
            z: tensor [*sample_shape, *batch_shape, num_datapoints_so_far]
        Returns: distribution with batch_shape [*sample_shape, *batch_shape] and event_shape []
        """
        raise NotImplementedError()

    def init_z(self, sample_shape=torch.Size([])):
        return torch.zeros(*sample_shape, *self.batch_shape, 0).long()

    def sample(self, sample_shape=torch.Size([])):
        z = self.next_dist(self.init_z(sample_shape)).sample()[..., None]
        for i in range(self.num_data - 1):
            z = torch.cat([z, self.next_dist(z).sample()[..., None]], dim=-1)
        assert tuple(z.shape) == (*sample_shape, *self.batch_shape, *self.event_shape)
        return z

    def log_prob(self, z):
        """
        Args:
            z: tensor [*sample_shape, *batch_shape, num_data]
        Returns: tensor [*sample_shape, *batch_shape]
        """
        sample_shape = z.shape[: -len(self.batch_shape) - 1]
        p0 = self.next_dist(self.init_z(sample_shape)).log_prob(z[..., 0])
        pi = [self.next_dist(z[..., :i]).log_prob(z[..., i]) for i in range(1, self.num_data)]
        p = torch.stack([p0, *pi]).sum(dim=0)
        assert tuple(p.shape) == (*sample_shape, *self.batch_shape)
        return p

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return torch.Size([self.num_data])

    def enumerate(self):
        if not hasattr(self, "_enumerate"):

            def get_z_by_max(n):
                if n == 1:
                    return [torch.Tensor([[0]]).long()]
                else:
                    z_by_max = [[] for _ in range(n)]
                    for old_max, old_z in enumerate(get_z_by_max(n - 1)):
                        for new_elem in range(old_max + 2):
                            new_max = max(old_max, new_elem)
                            new_zs = torch.cat(
                                [old_z, torch.ones(len(old_z), 1).long() * new_elem], dim=-1
                            )
                            z_by_max[new_max].append(new_zs)
                    z_by_max = [torch.cat(zs, dim=0) for zs in z_by_max]
                    return z_by_max

            self._enumerate = torch.cat(get_z_by_max(self.num_data), dim=0)
        return self._enumerate


class CRP(ClusterDist):
    def __init__(self, num_data):
        super().__init__(num_data)

    def next_dist(self, z):
        count = (z[..., None, :] == torch.arange(z.shape[-1] + 1)[:, None]).sum(dim=-1).float()
        n = (count > 0).sum(dim=-1, dtype=torch.long)
        probs_unnormalized = count + torch.nn.functional.one_hot(n, z.shape[-1] + 1)
        return torch.distributions.Categorical(probs=probs_unnormalized)


class MaskedSoftmaxClustering(ClusterDist):
    def __init__(self, logits):
        """
        Args:
            logits: tensor [*batch_shape, num_data, num_data]
        """
        super().__init__(logits.shape[-1], logits.shape[:-2])
        self.logits = logits

    def next_dist(self, z):
        """
        Args:
            z: tensor [*sample_shape, *batch_shape, num_datapoints_so_far]
        Returns: distribution with batch_shape [*sample_shape, *batch_shape] and event_shape []
        """
        assert tuple(z.shape[-len(self.batch_shape) - 1 : -1]) == tuple(self.batch_shape)
        sample_shape = z.shape[: -len(self.batch_shape) - 1]

        # count: tensor [*sample_shape *batch_shape, num_clusters] -- how many times has each cluster been used
        count = (
            (z[..., None, :] == torch.arange(self.logits.shape[-1])[:, None]).sum(dim=-1).float()
        )

        # n: tensor [*sample_shape, *batch_shape] -- how many observations total
        n = (count > 0).sum(dim=-1, dtype=torch.long)

        # hide: tensor [*sample_shape, *batch_shape, num_clusters] -- Whether to hide this cluster
        hide = (count + torch.nn.functional.one_hot(n, self.logits.shape[-1])) == 0

        # logits: tensor [*sample_shape, batch_shape, num_clusters] -- logits for the next datapoint
        logits = self.logits[..., z.shape[-1]] * 1
        logits = logits.repeat(*sample_shape, *logits.ndim * [1])

        logits[hide] = float("-inf")
        return torch.distributions.Categorical(logits=logits)


if __name__ == "__main__":
    batch_size = 7
    num_data = 10

    print("--- CRP ---")
    crp = CRP(num_data)
    print("Support size = ", len(crp.enumerate()))
    z = crp.sample([batch_size])
    print("z =", z)
    log_prob = crp.log_prob(z)
    print("log_prob =", log_prob)

    print("\n--- Softmax ---")
    softmax_clustering = MaskedSoftmaxClustering(logits=torch.randn(batch_size, num_data, num_data))
    print("Support size = ", len(softmax_clustering.enumerate()))
    z = softmax_clustering.sample()
    print("z =", z)
    log_prob = softmax_clustering.log_prob(z)
    print("log_prob =", log_prob)
