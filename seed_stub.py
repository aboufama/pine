"""gpu/seed_data.py exactly, with fetch() replaced by a local SOMA-style archive (LAFAN1 BVH as 40 actors)."""
import csv, glob, os, sys, tarfile
sys.path.insert(0, "gpu")
import seed_data as SD


def fake_fetch():
    os.makedirs(os.path.join(SD.RAW, "metadata"), exist_ok=True)
    src = sorted(glob.glob(os.path.join(os.environ["PINE_RAW"], "lafan1", "repo", "lafan1", "bvh", "*.bvh")))
    tar_path = os.path.join(SD.RAW, "bones_seed_soma_proportional.tar"); rows = []
    with tarfile.open(tar_path, "w") as tf:
        for i, f in enumerate(src):
            name = f"soma_proportional/bvh/take_{i:04d}.bvh"; tf.add(f, arcname=name)
            rows.append(dict(filename=name, actor_uid=f"actor{i % 40:02d}", is_mirror="false"))
    with open(os.path.join(SD.RAW, "metadata", "seed_metadata.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, ["filename", "actor_uid", "is_mirror"]); w.writeheader(); w.writerows(rows)
    return dict(archives=[os.path.basename(tar_path)], local=[tar_path])


SD.fetch = fake_fetch
if __name__ == "__main__":
    rc = SD.main()
    print(open(os.path.join(SD.OUT, "SEED_OK" if rc == 0 else "SEED_FAILED")).read()[:600])
    sys.exit(rc)
