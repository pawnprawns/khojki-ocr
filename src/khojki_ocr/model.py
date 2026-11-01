"""163,736-parameter multi-head character CNN and its 1,416-label decoder."""

from pathlib import Path
import hashlib, json
import torch
from torch import nn
from torch.nn import functional as F

MARKS = [chr(c) for c in (0x11234, 0x11235, 0x11236, 0x11237)]


def is_consonant(cp):
    return 0x11208 <= cp <= 0x1122B and cp != 0x11212 or cp == 0x1123F


def is_vowel(cp):
    return 0x11200 <= cp <= 0x11207 or cp == 0x11240


class Schema:

    def __init__(self, labels):
        self.labels = labels
        self.identities = sorted({s[0] for s in labels})
        identity_index = {s: i for i, s in enumerate(self.identities)}
        self.base = []
        self.vowel = []
        self.marks = []
        self.vm = []
        self.mm = []
        for s in labels:
            cp = ord(s[0])
            tail = s[1:]
            cons = is_consonant(cp)
            vow = is_vowel(cp)
            vs = [ord(c) - 0x1122C + 1 for c in tail if 0x1122C <= ord(c) <= 0x11233]
            assert len(vs) <= 1
            self.base.append(identity_index[s[0]])
            self.vowel.append(vs[0] if vs else 0)
            self.marks.append([int(m in tail) for m in MARKS])
            self.vm.append(float(cons))
            self.mm.append([float(cons or vow), float(cons), float(cons), float(cons)])
        self.base = torch.tensor(self.base)
        self.vowel = torch.tensor(self.vowel)
        self.marks = torch.tensor(self.marks, dtype=torch.float32)
        self.vm = torch.tensor(self.vm)
        self.mm = torch.tensor(self.mm)
        assert len(
            {
                tuple(
                    [int(self.base[i]), int(self.vowel[i])]
                    + self.marks[i].int().tolist()
                )
                for i in range(len(labels))
            }
        ) == len(labels)
        assert [
            self.compose(int(b), int(v), m.tolist())
            for b, v, m in zip(self.base, self.vowel, self.marks)
        ] == labels

    def compose(self, b, v, marks):
        s = self.identities[b]
        if marks[2]:
            s += MARKS[2]
        if v:
            s += chr(0x1122C + v - 1)
        for k in (0, 1, 3):
            if marks[k]:
                s += MARKS[k]
        return s

    def loss(self, out, y):
        b, v, m = out
        bl = F.cross_entropy(b, self.base[y])
        vl = (
            F.cross_entropy(v, self.vowel[y], reduction="none") * self.vm[y]
        ).sum() / self.vm[y].sum().clamp_min(1)
        ml = (
            F.binary_cross_entropy_with_logits(m, self.marks[y], reduction="none")
            * self.mm[y]
        ).sum() / self.mm[y].sum().clamp_min(1)
        return bl + vl + ml

    def scores(self, out):
        b, v, m = out
        scores = (
            F.log_softmax(b, 1)[:, self.base]
            + F.log_softmax(v, 1)[:, self.vowel] * self.vm
        )
        pos = F.logsigmoid(m)
        neg = F.logsigmoid(-m)
        scores += pos @ (self.marks * self.mm).T + neg @ ((1 - self.marks) * self.mm).T
        return scores


class CNN(nn.Module):

    def __init__(self, schema):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveMaxPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(1152, 128),
            nn.ReLU(),
        )
        self.base = nn.Linear(128, len(schema.identities))
        self.vowel = nn.Linear(128, 9)
        self.marks = nn.Linear(128, 4)

    def forward(self, x):
        z = self.encoder(x)
        return (self.base(z), self.vowel(z), self.marks(z))


ASSETS = Path(__file__).resolve().parent / "assets"


def load_model():
    """Verify bundled assets before loading a CPU-only, weights-only checkpoint."""
    config = json.loads((ASSETS / "model.json").read_text(encoding="utf-8"))
    for name, expected in config["sha256"].items():
        if hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Bundled model asset checksum mismatch: " + name)
    schema = Schema(json.loads((ASSETS / "labels.json").read_text(encoding="utf-8")))
    model = CNN(schema)
    model.load_state_dict(
        torch.load(ASSETS / "classifier.pt", map_location="cpu", weights_only=True)[
            "state_dict"
        ]
    )
    model.eval()
    return model, schema, config["views"], config
