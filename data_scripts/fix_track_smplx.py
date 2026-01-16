import os

ehmx_dir = '/home/szj/err_empty_syn/ytb/EHM-X/shots_ehmx'

if not os.path.exists(ehmx_dir):
    print(f"Directory not found: {ehmx_dir}")
    exit(1)

print(f"Scanning {ehmx_dir}...")

for root, dirs, files in os.walk(ehmx_dir):
    if 'track_smplx.pkl' in files:
        old_path = os.path.join(root, 'track_smplx.pkl')
        new_path = os.path.join(root, 'optim_tracking_ehm.pkl')
        
        if os.path.exists(new_path):
            print(f"Skipping {old_path} -> {new_path} (Target already exists)")
        else:
            print(f"Renaming {old_path} -> {new_path}")
            os.rename(old_path, new_path)
