"""PredRNN backbone and wind-map metrics for the regional forecast comparison."""
import torch
from torch import nn
from torch.nn import functional as F

from experiment import FIELDS, LEADS, MemoryCell


class DecoupledCell(MemoryCell):
    @staticmethod
    def write_memory(gates, previous):
        forget, write, value = gates.chunk(3, dim=1)
        increment = write.sigmoid() * value.tanh()
        return (forget + 1).sigmoid() * previous + increment, increment

    def forward(self, x, h, c, m):
        c, dc = self.write_memory(self.temporal(torch.cat((x, h), 1)), c)
        m, dm = self.write_memory(self.spatial(torch.cat((x, m), 1)), m)
        gate = self.output_gate(torch.cat((x, h, c, m), 1)).sigmoid()
        return gate * self.read(torch.cat((c, m), 1)).tanh(), c, m, dc, dm


class PaperPredRNN(nn.Module):
    """Regional ST-LSTM with decoupled write increments and reverse sampling."""
    def __init__(self, hidden=8):
        super().__init__()
        self.hidden = hidden
        self.cells = nn.ModuleList([DecoupledCell(len(FIELDS), hidden), DecoupledCell(hidden, hidden)])
        self.readout = nn.Conv2d(hidden, len(FIELDS), 1)
        self.adapter = nn.Conv2d(hidden, hidden, 1, bias=False)

    def sequence(self, history, future=None, progress=1.):
        n, length, _, height, width = history.shape
        memory = history.new_zeros((n, self.hidden, height, width))
        h, c = [memory, memory], [memory, memory]
        outputs, penalties = [], []
        for step in range(length + len(LEADS) - 1):
            if step == 0:
                frame = history[:, 0]
            elif future is None:
                frame = history[:, step] if step < length else generated
            else:
                observed = history[:, step] if step < length else future[:, step-length]
                prob = .5 + .5*progress if step < length else .5*(1-progress)
                mask = torch.rand(n, 1, 1, 1, device=history.device) < prob
                frame = torch.where(mask, observed, generated)
            x = frame
            for i, cell in enumerate(self.cells):
                h[i], c[i], memory, dc, dm = cell(x, h[i], c[i], memory)
                x = h[i]
                if future is not None:
                    a, b = self.adapter(dc).flatten(2), self.adapter(dm).flatten(2)
                    penalties.append(F.cosine_similarity(a, b, dim=2).abs().mean())
            generated = frame + self.readout(x)
            if step >= length - 1:
                outputs.append(generated)
        penalty = torch.stack(penalties).mean() if penalties else history.new_zeros(())
        return torch.stack(outputs, dim=1), penalty

    def forward(self, history):
        return self.sequence(history)[0]


class WindMapMetrics:
    """SSIM and LPIPS on aligned 500 hPa speed images; same scaling for all models."""
    def __init__(self, max_speed):
        import lpips
        self.network = lpips.LPIPS(net="squeeze", version="0.1", verbose=False).eval()
        self.max_speed = max_speed

    def images(self, normalized, mean, std):
        wind = (normalized*std + mean)[:, :, :2, 1:-1, 1:-1]
        return (wind.norm(dim=2, keepdim=True)/self.max_speed).clamp(0, 1)

    @staticmethod
    def rgb(x):
        # Same aspect ratio as the native 3 x 5 map. LPIPS expects three channels.
        gray = F.interpolate(x, size=(48, 80), mode="bilinear", align_corners=False)
        return gray.expand(-1, 3, -1, -1).contiguous().mul(2).sub(1)

    @torch.no_grad()
    def score(self, pred, truth, mean, std):
        n, leads = pred.shape[:2]
        x = self.images(pred, mean, std).flatten(0, 1)
        y = self.images(truth, mean, std).flatten(0, 1)
        def avg(z):
            return F.avg_pool2d(z, 3, stride=1)
        mx, my = avg(x), avg(y)
        vx, vy, cov = avg(x*x)-mx*mx, avg(y*y)-my*my, avg(x*y)-mx*my
        ssim = ((2*mx*my + .01**2)*(2*cov + .03**2) /
                ((mx.square()+my.square()+.01**2)*(vx+vy+.03**2)))
        perceptual = [self.network(self.rgb(a), self.rgb(b)).flatten()
                      for a, b in zip(x.split(32), y.split(32))]
        return ssim.mean((1, 2, 3)).reshape(n, leads).mean(0), torch.cat(perceptual).reshape(n, leads).mean(0)
