import json
p = "lfm350m-vi-cpt-r3/adapter_config.json"
c = json.load(open(p))
c["inference_mode"] = False
json.dump(c, open(p, "w"), indent=2)
print("inference_mode:", c["inference_mode"])
