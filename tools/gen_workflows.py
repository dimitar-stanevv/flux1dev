#!/usr/bin/env python3
"""Generate (and validate) the two bundled ComfyUI workflows.

Hand-writing ComfyUI's UI-format JSON is how you end up with a graph that
opens with silently missing wires. This builds both graphs from one shared
spec, then checks every link from both ends before writing anything:

  * every input's `link` id exists in `links` and points at that node + slot
  * every id in an output's `links` list has that node + slot as its origin

Run from the repo root:

    python tools/gen_workflows.py

Output: workflows/flux_dev_power_lora.json, workflows/flux_dev_lora_chain.json
"""

import json
import os
import sys

# Placeholder only — pick your actual LoRA from the dropdown once
# provision.py has pulled it onto the volume via the LORAS env var.
PLACEHOLDER_LORA = "my_lora.safetensors"

PROMPT = (
    "TOK, a cinematic portrait photograph, 85mm lens, soft window light, "
    "shallow depth of field, natural skin texture"
)

# Bypassed (mode 4) so the graph runs on a fresh volume with no LoRA present.
BYPASS = 4


class Graph(object):
    def __init__(self):
        self.nodes = []
        self.links = []
        self._node_id = 0
        self._link_id = 0

    def add(self, type_name, pos, size, widgets=None, inputs=(), outputs=(),
            mode=0, title=None):
        self._node_id += 1
        node = {
            "id": self._node_id,
            "type": type_name,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": 0,
            "mode": mode,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
            "outputs": [
                {"name": n, "type": t, "links": [], "slot_index": i}
                for i, (n, t) in enumerate(outputs)
            ],
            "properties": {"Node name for S&R": type_name},
            "widgets_values": widgets if widgets is not None else [],
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    def connect(self, src, src_slot, dst, dst_slot):
        out = src["outputs"][src_slot]
        inp = dst["inputs"][dst_slot]
        if out["type"] != inp["type"]:
            raise ValueError(
                "type mismatch: %s.%s (%s) -> %s.%s (%s)"
                % (src["type"], out["name"], out["type"],
                   dst["type"], inp["name"], inp["type"])
            )
        self._link_id += 1
        lid = self._link_id
        self.links.append([lid, src["id"], src_slot, dst["id"], dst_slot, out["type"]])
        out["links"].append(lid)
        inp["link"] = lid
        return lid

    def _order(self):
        """Topological order, so ComfyUI's own sort has nothing to fix up."""
        incoming = {n["id"]: set() for n in self.nodes}
        outgoing = {n["id"]: set() for n in self.nodes}
        for _, src, _, dst, _, _ in self.links:
            incoming[dst].add(src)
            outgoing[src].add(dst)

        ready = sorted(nid for nid, deps in incoming.items() if not deps)
        order = 0
        while ready:
            nid = ready.pop(0)
            self._by_id(nid)["order"] = order
            order += 1
            for nxt in sorted(outgoing[nid]):
                incoming[nxt].discard(nid)
                if not incoming[nxt]:
                    ready.append(nxt)
                    ready.sort()
        if order != len(self.nodes):
            raise ValueError("graph has a cycle")

    def _by_id(self, nid):
        for n in self.nodes:
            if n["id"] == nid:
                return n
        raise KeyError(nid)

    def build(self):
        self._order()
        return {
            "last_node_id": self._node_id,
            "last_link_id": self._link_id,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate(graph, name):
    errors = []
    by_id = {n["id"]: n for n in graph["nodes"]}
    links = {}
    for link in graph["links"]:
        if len(link) != 6:
            errors.append("malformed link entry: %r" % (link,))
            continue
        lid, src, src_slot, dst, dst_slot, ltype = link
        if lid in links:
            errors.append("duplicate link id %s" % lid)
        links[lid] = (src, src_slot, dst, dst_slot, ltype)

    for lid, (src, src_slot, dst, dst_slot, ltype) in links.items():
        if src not in by_id:
            errors.append("link %s originates at unknown node %s" % (lid, src))
        if dst not in by_id:
            errors.append("link %s targets unknown node %s" % (lid, dst))
        if src in by_id and not (0 <= src_slot < len(by_id[src]["outputs"])):
            errors.append("link %s: node %s has no output slot %s" % (lid, src, src_slot))
        if dst in by_id and not (0 <= dst_slot < len(by_id[dst]["inputs"])):
            errors.append("link %s: node %s has no input slot %s" % (lid, dst, dst_slot))

    for node in graph["nodes"]:
        for slot, inp in enumerate(node["inputs"]):
            lid = inp.get("link")
            if lid is None:
                errors.append("%s (id %s) input '%s' is not connected"
                              % (node["type"], node["id"], inp["name"]))
                continue
            if lid not in links:
                errors.append("%s (id %s) input '%s' references missing link %s"
                              % (node["type"], node["id"], inp["name"], lid))
                continue
            src, src_slot, dst, dst_slot, ltype = links[lid]
            if dst != node["id"] or dst_slot != slot:
                errors.append(
                    "link %s should target %s slot %s but the link says %s slot %s"
                    % (lid, node["id"], slot, dst, dst_slot))
            if ltype != inp["type"]:
                errors.append("link %s type %s but input '%s' wants %s"
                              % (lid, ltype, inp["name"], inp["type"]))

        for slot, out in enumerate(node["outputs"]):
            for lid in out.get("links") or []:
                if lid not in links:
                    errors.append("%s (id %s) output '%s' lists missing link %s"
                                  % (node["type"], node["id"], out["name"], lid))
                    continue
                src, src_slot, dst, dst_slot, ltype = links[lid]
                if src != node["id"] or src_slot != slot:
                    errors.append(
                        "link %s should originate at %s slot %s but the link says %s slot %s"
                        % (lid, node["id"], slot, src, src_slot))
                if ltype != out["type"]:
                    errors.append("link %s type %s but output '%s' is %s"
                                  % (lid, ltype, out["name"], out["type"]))

    if graph["last_node_id"] != max(by_id):
        errors.append("last_node_id %s but highest node id is %s"
                      % (graph["last_node_id"], max(by_id)))
    if links and graph["last_link_id"] != max(links):
        errors.append("last_link_id %s but highest link id is %s"
                      % (graph["last_link_id"], max(links)))

    if errors:
        print("FAIL %s" % name)
        for err in errors:
            print("  - %s" % err)
        return False
    print("ok   %s: %d nodes, %d links" % (name, len(graph["nodes"]), len(links)))
    return True


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------

def build(lora_stage):
    """Shared Flux graph. `lora_stage(g, unet, dual_clip)` returns
    (model_node, model_slot, clip_node, clip_slot)."""
    g = Graph()

    unet = g.add("UNETLoader", (40, 40), (330, 82),
                 widgets=["flux1-dev.safetensors", "default"],
                 outputs=[("MODEL", "MODEL")])
    dual = g.add("DualCLIPLoader", (40, 180), (330, 106),
                 widgets=["t5xxl_fp16.safetensors", "clip_l.safetensors", "flux"],
                 outputs=[("CLIP", "CLIP")])
    vae = g.add("VAELoader", (40, 340), (330, 58),
                widgets=["ae.safetensors"],
                outputs=[("VAE", "VAE")])

    model_node, model_slot, clip_node, clip_slot = lora_stage(g, unet, dual)

    text = g.add("CLIPTextEncode", (900, 40), (420, 200),
                 widgets=[PROMPT],
                 inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 title="Prompt")
    guidance = g.add("FluxGuidance", (1380, 40), (300, 58),
                     widgets=[3.5],
                     inputs=[("conditioning", "CONDITIONING")],
                     outputs=[("CONDITIONING", "CONDITIONING")])
    guider = g.add("BasicGuider", (1380, 150), (300, 66),
                   inputs=[("model", "MODEL"), ("conditioning", "CONDITIONING")],
                   outputs=[("GUIDER", "GUIDER")])

    latent = g.add("EmptySD3LatentImage", (900, 300), (330, 106),
                   widgets=[1024, 1024, 1],
                   outputs=[("LATENT", "LATENT")])
    noise = g.add("RandomNoise", (900, 450), (330, 82),
                  widgets=[0, "randomize"],
                  outputs=[("NOISE", "NOISE")])
    sampler = g.add("KSamplerSelect", (900, 570), (330, 58),
                    widgets=["euler"],
                    outputs=[("SAMPLER", "SAMPLER")])
    scheduler = g.add("BasicScheduler", (900, 670), (330, 106),
                      widgets=["simple", 20, 1.0],
                      inputs=[("model", "MODEL")],
                      outputs=[("SIGMAS", "SIGMAS")])

    advanced = g.add("SamplerCustomAdvanced", (1380, 280), (340, 126),
                     inputs=[("noise", "NOISE"), ("guider", "GUIDER"),
                             ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
                             ("latent_image", "LATENT")],
                     outputs=[("output", "LATENT"), ("denoised_output", "LATENT")])
    decode = g.add("VAEDecode", (1780, 40), (260, 50),
                   inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")])
    save = g.add("SaveImage", (1780, 150), (460, 500),
                 widgets=["flux/flux"],
                 inputs=[("images", "IMAGE")])

    g.connect(model_node, model_slot, guider, 0)
    g.connect(model_node, model_slot, scheduler, 0)
    g.connect(clip_node, clip_slot, text, 0)
    g.connect(text, 0, guidance, 0)
    g.connect(guidance, 0, guider, 1)
    g.connect(noise, 0, advanced, 0)
    g.connect(guider, 0, advanced, 1)
    g.connect(sampler, 0, advanced, 2)
    g.connect(scheduler, 0, advanced, 3)
    g.connect(latent, 0, advanced, 4)
    g.connect(advanced, 0, decode, 0)
    g.connect(vae, 0, decode, 1)
    g.connect(decode, 0, save, 0)
    return g


def power_lora_stage(g, unet, dual):
    """One rgthree Power Lora Loader holding an arbitrary list of LoRAs."""
    node = g.add(
        "Power Lora Loader (rgthree)", (450, 40), (400, 240),
        widgets=[
            None,
            {"type": "PowerLoraLoaderHeaderWidget"},
            {"on": False, "lora": PLACEHOLDER_LORA, "strength": 1, "strengthTwo": None},
            {"on": False, "lora": "None", "strength": 1, "strengthTwo": None},
            {"on": False, "lora": "None", "strength": 1, "strengthTwo": None},
            "",
        ],
        inputs=[("model", "MODEL"), ("clip", "CLIP")],
        outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP")],
        title="Power Lora Loader (rgthree)",
    )
    node["properties"]["Show Strengths"] = "Single Strength"
    g.connect(unet, 0, node, 0)
    g.connect(dual, 0, node, 1)
    return node, 0, node, 1


def chain_lora_stage(g, unet, dual):
    """Three stock LoraLoaderModelOnly nodes in series. No custom nodes.

    Bypassed so the graph runs on a fresh volume; Ctrl+B re-enables one once
    you've picked a real LoRA in its dropdown.
    """
    upstream, slot = unet, 0
    for i, strength in enumerate((1.0, 0.0, 0.0)):
        node = g.add("LoraLoaderModelOnly", (450, 40 + i * 160), (340, 82),
                     widgets=[PLACEHOLDER_LORA, strength],
                     inputs=[("model", "MODEL")],
                     outputs=[("MODEL", "MODEL")],
                     mode=BYPASS,
                     title="LoRA %d (bypassed — Ctrl+B to enable)" % (i + 1))
        g.connect(upstream, slot, node, 0)
        upstream, slot = node, 0
    # CLIP goes straight through: these LoRAs carry no text-encoder weights.
    return upstream, slot, dual, 0


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "workflows")
    os.makedirs(out_dir, exist_ok=True)

    targets = [
        ("flux_dev_power_lora.json", power_lora_stage),
        ("flux_dev_lora_chain.json", chain_lora_stage),
    ]

    ok = True
    built = []
    for name, stage in targets:
        graph = build(stage).build()
        if not validate(graph, name):
            ok = False
            continue
        built.append((name, graph))

    if not ok:
        return 1

    for name, graph in built:
        path = os.path.join(out_dir, name)
        with open(path, "w") as fh:
            json.dump(graph, fh, indent=2)
            fh.write("\n")
        print("wrote %s" % os.path.relpath(path, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
