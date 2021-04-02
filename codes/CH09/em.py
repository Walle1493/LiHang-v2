import argparse


class EM():
    def __init__(self, y, init_prob, epsilon=1e-5, maxiter=10):
        self.obs = y # observable variable
        self.pi, self.p, self.q = init_prob # estimated value
        self.epsilon = epsilon
        self.maxiter = maxiter
        self.early_stop = False
        
    def e_step(self, j):    # j: observable num
        # fromula (9.5): calculate mu_j^(i+1)
        # numerator and denominator left part
        mu1 = self.pi * (self.p ** self.obs[j]) * ((1 - self.p) ** (1 - self.obs[j]))
        # denominator right part
        mu2 = (1 - self.pi) * (self.q ** self.obs[j]) * ((1 - self.q) ** (1 - self.obs[j]))
        # mu_j^(i+1)
        return mu1 / (mu1 + mu2)
    
    def m_step(self):
        obs_len = len(self.obs)
        mu = [self.e_step(j) for j in range(obs_len)]
        # formula (9.6), (9.7), (9.8): calculate pi, p, q
        pi = 1 / obs_len * sum(mu)
        p = sum([mu[j] * self.obs[j] for j in range(obs_len)]) / sum([mu[j] for j in range(obs_len)])
        q = sum([(1 - mu[j]) * self.obs[j] for j in range(obs_len)]) / sum([(1 - mu[j]) for j in range(obs_len)])
        # early stop iteration when diff<epsilon
        if abs(pi - self.pi) < self.epsilon and abs(p - self.p) < self.epsilon and abs(q - self.q) < self.epsilon:
            self.early_stop = True
        # update estimated values
        self.pi = pi
        self.p = p
        self.q = q

    def fit(self):
        print("Before EM algorithm: pi={:.5f}, p={:.5f}, q={:.5f}".format(self.pi, self.p, self.q))
        for epoch in range(1, self.maxiter + 1):
            # self.e_step()
            self.m_step()
            print("After {} epoch: pi={:.5f}, p={:.5f}, q={:.5f}".format(epoch, self.pi, self.p, self.q))
            if self.early_stop:
                print("Already converged!")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi", type=float, default=0.45)
    parser.add_argument("--p", type=float, default=0.65)
    parser.add_argument("--q", type=float, default=0.55)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=float, default=10)
    args = parser.parse_args()

    # construct initial observable values and initial probability
    y = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    init_prob = [args.pi, args.p, args.q]

    # train
    em = EM(y=y, init_prob=init_prob, epsilon=args.epsilon, maxiter=args.maxiter)
    em.fit()

    # print final estimated values
    print("Finally: pi={:.5f}, p={:.5f}, q={:.5f}".format(em.pi, em.p, em.q))
