default_config = {
    "machine": {
        "batch_type": "Slurm",
        "local_root": "./",
        "remote_root": "/home/yuanf/4_workfplace/",
        "context_type": "SSHContext",
        "remote_profile": {
            "hostname": "cheaha.rc.uab.edu",
            "username": "yuanf",
            "port": 22,
        },
    },
    "resources": {
        "number_node": 1,
        "cpu_per_node": 8,
        "gpu_per_node": 1,
        "queue_name": "amperenodes",
        "group_size": 8,
        "prepend_script": [
            "source ~/deepmd-kit-3.0.1/bin/activate ~/deepmd-kit-3.0.1/"
        ],
    },
}
