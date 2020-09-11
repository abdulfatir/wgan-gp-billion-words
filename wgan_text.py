import torch
import numpy as np
from tqdm import tqdm

import networks
import language_helpers

# Dataset iterator
def inf_train_gen(lines, charmap, batch_size):
    while True:
        np.random.shuffle(lines)
        for i in range(0, len(lines)-batch_size+1, batch_size):
            yield np.array(
                [[charmap[c] for c in l] for l in lines[i:i+batch_size]], 
                dtype='int32')

def GP(D, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1).to(real_data.device)
    alpha = alpha.expand_as(real_data)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad_()
    dinterpolates = D(interpolates)

    gradients = torch.autograd.grad(outputs=dinterpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(dinterpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean(0)
    return gradient_penalty

def train(G, D, dgen, iters, seq_len, batch_size, critic_iters, cmaplen, inv_charmap, ngrams, true_char_ngram_lms, device=torch.device('cuda:0')):
    G = G.to(device)
    D = D.to(device)
    g_optim = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optim = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    x_real_oh = torch.zeros((batch_size, seq_len, cmaplen)).to(device)
    for i in range(iters):
        # Discriminator Iterations
        for p in D.parameters():
            p.requires_grad = True
        for _ in range(critic_iters):
            D.zero_grad()
            x_real = torch.LongTensor(next(dgen)).to(device)
            x_real = x_real.unsqueeze(-1)
            x_real_oh.zero_()
            x_real_oh.scatter_(2, x_real, 1)
            with torch.no_grad():
                z = torch.randn(batch_size, 128).to(device)
                x_fake = G(z)
            d_real = D(x_real_oh)
            d_fake = D(x_fake)
            gp = GP(D, x_real_oh, x_fake)
            d_loss = torch.mean(d_fake) - torch.mean(d_real) + 10. * gp
            d_loss.backward()
            d_optim.step()
        # Generator Iteration
        for p in D.parameters():
            p.requires_grad = False
        G.zero_grad()
        z = torch.randn(batch_size, 128).to(device)
        x_fake = G(z)
        d_fake = D(x_fake)
        g_loss = -torch.mean(d_fake)
        g_loss.backward()
        g_optim.step()

        if i % 1000 == 0:
            # Generate Samples
            samples = []
            for _ in range(10):
                with torch.no_grad():
                    z = torch.randn(batch_size, 128).to(device)
                    x_fake = G(z)
                x_fake = np.argmax(x_fake.cpu().numpy(), axis=2)
                x_fake_decoded = []
                for k in range(len(x_fake)):
                    decoded = []
                    for l in range(len(x_fake[k])):
                        decoded.append(inv_charmap[x_fake[k][l]])
                    x_fake_decoded.append(tuple(decoded))
                samples.extend(x_fake_decoded)
            # Compute JS
            js_n = []
            for m, n in enumerate(ngrams):
                lm = language_helpers.NgramLanguageModel(n, samples, tokenize=False)
                js_n.append(lm.js_with(true_char_ngram_lms[m]))
            with open('results/samples_{}.txt'.format(i), 'w') as f:
                for s in samples:
                    s = ''.join(s)
                    f.write(s + '\n')
            torch.save(G.state_dict(), f'results/wgan_G_{i}.pth')
            torch.save(G.state_dict(), f'results/wgan_D_{i}.pth')
        if i >= 0:
            print(f'Iter {i}| JS-4:{js_n[0]:.4f}, g: {g_loss.item():.4f}, d: {d_loss.item():.4f}')
    return G, D
            
def main():
    data_dir = './data/1-billion-word-language-modeling-benchmark-r13output/'
    batch_size = 64
    iters = 100000
    seq_len = 32
    dim = 512
    critic_iters = 5
    gp_scale = 10
    max_n_examples = 10000000
    sn = False

    lines, charmap, inv_charmap = language_helpers.load_dataset(
    max_length=seq_len,
    max_n_examples=max_n_examples,
    data_dir=data_dir)
    dgen = inf_train_gen(lines, charmap, batch_size)
    G = networks.TextGenerator(len(charmap))
    D = networks.TextDiscriminator(len(charmap), sn=sn)
    ngrams = [4]
    true_char_ngram_lms = [language_helpers.NgramLanguageModel(n, lines[10*batch_size:], tokenize=False) for n in ngrams]
    validation_char_ngram_lms = [language_helpers.NgramLanguageModel(n, lines[:10*batch_size], tokenize=False) for n in ngrams]
    for i, n in enumerate(ngrams):
        print("validation set JSD for n={}: {:.4f}".format(n, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
    true_char_ngram_lms = [language_helpers.NgramLanguageModel(n, lines, tokenize=False) for n in ngrams]
    train(G, D, dgen, iters, seq_len, batch_size, critic_iters, len(charmap), inv_charmap, ngrams, true_char_ngram_lms)


if __name__ == '__main__':
    main()
