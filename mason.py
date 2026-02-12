"""
Simplified mason launcher for Beaker jobs.
Adapted from https://github.com/allenai/open-instruct/blob/main/mason.py
Stripped of open-instruct-specific logic (dataset caching, GCP uploads, etc.)
"""

import argparse
import os
import secrets
import string

import backoff
import beaker
import requests
from rich.console import Console
from rich.text import Text

console = Console()

WEKA_CLUSTERS = [
    "ai2/jupiter", "ai2/saturn", "ai2/titan", "ai2/neptune",
    "ai2/ceres", "ai2/triton", "ai2/rhea",
]
GCP_CLUSTERS = ["ai2/augusta"]
INTERCONNECT_CLUSTERS = ["ai2/jupiter", "ai2/ceres", "ai2/titan", "ai2/augusta"]

DEFAULT_ENV_VARS = {
    "NCCL_DEBUG": "ERROR",
    "VLLM_LOGGING_LEVEL": "WARNING",
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
}


def parse_beaker_dataset(dataset_str: str) -> dict[str, str]:
    splt = dataset_str.split(":")
    if len(splt) != 2:
        raise argparse.ArgumentTypeError(f"Invalid dataset format: {dataset_str}. Expected 'mount_path:beaker_id'")
    return {"mount_path": splt[0], "beaker": splt[1]}


def parse_env_var(env_var_str: str) -> dict[str, str]:
    if "=" not in env_var_str:
        raise argparse.ArgumentTypeError(f"Environment variable must be in format 'name=value', got: {env_var_str}")
    name, value = env_var_str.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("Environment variable name cannot be empty")
    return {"name": name, "value": value}


def generate_id(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


global_wandb_id = generate_id()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster", type=str, nargs="+", required=True)
    parser.add_argument("--hostname", type=str, nargs="+", default=None)
    parser.add_argument("--max_retries", type=int, default=0)
    parser.add_argument("--budget", type=str, required=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--shared_memory", type=str, default="10.24gb")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--image", type=str, default="ai2/cuda11.8-cudnn8-dev-ubuntu20.04")
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--beaker_datasets", nargs="*", type=parse_beaker_dataset, default=[])
    parser.add_argument("--description", type=str, default="Beaker-Mason job.")
    parser.add_argument("--task_name", type=str, default="beaker_mason")
    parser.add_argument("--priority", type=str, default="normal")
    parser.add_argument("--preemptible", action="store_true")
    parser.add_argument("--pure_docker_mode", action="store_true")
    parser.add_argument("--no-host-networking", action="store_true")
    parser.add_argument(
        "--env", type=parse_env_var, action="append", default=[],
        help="Additional env vars in format 'name=value'. Can be specified multiple times.",
    )
    parser.add_argument(
        "--secret", type=parse_env_var, action="append", default=[],
        help="Additional secret env vars in format 'name=value'. Can be specified multiple times.",
    )
    parser.add_argument("--timeout", type=str, default=None)

    mason_args, command_args = parser.parse_known_args()
    commands = parse_commands(command_args)
    return mason_args, commands


def parse_commands(command_args: list[str]) -> list[list[str]]:
    if command_args[0] != "--":
        raise Exception(
            "Please separate the command with ' -- ', like "
            "`python mason.py [mason-args] -- your_command [args]`."
        )
    commands = []
    command = []
    for item in command_args:
        if item == "--":
            if command:
                commands.append(command)
                command = []
        else:
            command.append(item)
    if command:
        commands.append(command)
    return commands


def get_env_vars(
    pure_docker_mode: bool,
    cluster: list[str],
    beaker_secrets: list[str],
    whoami: str,
    num_nodes: int,
    additional_env_vars: list[dict[str, str]],
    additional_secrets: list[dict[str, str]],
):
    additional_env_var_names = {var["name"] for var in additional_env_vars}
    env_vars = [
        beaker.BeakerEnvVar(name=name, value=value)
        for name, value in DEFAULT_ENV_VARS.items()
        if name not in additional_env_var_names
    ]
    env_vars.extend(
        [beaker.BeakerEnvVar(name=env_var["name"], value=env_var["value"]) for env_var in additional_env_vars]
    )
    env_vars.extend(
        [beaker.BeakerEnvVar(name=secret["name"], secret=secret["value"]) for secret in additional_secrets]
    )

    useful_secrets = ["HF_TOKEN", "WANDB_API_KEY", "BEAKER_TOKEN"]
    for useful_secret in useful_secrets:
        if f"{whoami}_{useful_secret}" in beaker_secrets:
            env_vars.append(beaker.BeakerEnvVar(name=useful_secret, secret=f"{whoami}_{useful_secret}"))
        elif useful_secret in beaker_secrets:
            env_vars.append(beaker.BeakerEnvVar(name=useful_secret, secret=useful_secret))

    if not pure_docker_mode:
        env_vars.extend([beaker.BeakerEnvVar(name="PATH", value=os.getenv("PATH"))])

    if all(c in WEKA_CLUSTERS for c in cluster):
        env_vars.extend([
            beaker.BeakerEnvVar(name="HF_HOME", value="/weka/oe-adapt-default/allennlp/.cache/huggingface"),
            beaker.BeakerEnvVar(name="HF_DATASETS_CACHE", value="/weka/oe-adapt-default/allennlp/.cache/huggingface"),
            beaker.BeakerEnvVar(name="HF_HUB_CACHE", value="/weka/oe-adapt-default/allennlp/.cache/hub"),
        ])
        if num_nodes > 1:
            env_vars.extend([
                beaker.BeakerEnvVar(name="NCCL_SOCKET_IFNAME", value="ib"),
                beaker.BeakerEnvVar(name="NCCL_IB_HCA", value="^=mlx5_bond_0"),
            ])

    return env_vars


def get_datasets(beaker_datasets, cluster: list[str]):
    res = []
    if all(c in WEKA_CLUSTERS for c in cluster):
        res = [
            beaker.BeakerDataMount(
                source=beaker.BeakerDataSource(weka="oe-adapt-default"), mount_path="/weka/oe-adapt-default"
            ),
            beaker.BeakerDataMount(
                source=beaker.BeakerDataSource(weka="oe-training-default"), mount_path="/weka/oe-training-default"
            ),
        ]
    for beaker_dataset in beaker_datasets:
        to_append = beaker.BeakerDataMount(
            source=beaker.BeakerDataSource(beaker=beaker_dataset["beaker"]), mount_path=beaker_dataset["mount_path"]
        )
        res.append(to_append)
    return res


def make_internal_command(command: list[str], args: argparse.Namespace) -> str:
    for i in range(len(command)):
        if "</" in command[i]:
            command[i] = f"'{command[i]}'"
    for idx in range(len(command)):
        if "{" in command[idx]:
            command[idx] = "'" + command[idx] + "'"

    setup_commands = ""
    if not args.pure_docker_mode:
        setup_commands = f"cd {os.getcwd()} && "

    full_command = setup_commands + " ".join(command)
    console.log("Full command:")
    print(full_command)
    return full_command


def make_task_spec(args, full_command: str, i: int, beaker_secrets: list[str], whoami: str):
    if args.hostname is not None:
        constraints = beaker.BeakerConstraints(hostname=args.hostname)
    else:
        constraints = beaker.BeakerConstraints(cluster=args.cluster)
    spec = beaker.BeakerTaskSpec(
        name=f"{args.task_name}__{i}",
        image=beaker.BeakerImageSource(beaker=args.image),
        command=["/bin/bash", "-c"],
        arguments=[full_command],
        result=beaker.BeakerResultSpec(path="/output"),
        datasets=get_datasets(args.beaker_datasets, args.cluster),
        context=beaker.BeakerTaskContext(
            priority=beaker.BeakerJobPriority[args.priority], preemptible=args.preemptible
        ),
        constraints=constraints,
        env_vars=get_env_vars(
            args.pure_docker_mode, args.cluster, beaker_secrets, whoami,
            args.num_nodes, args.env, args.secret,
        ),
        resources=beaker.BeakerTaskResources(gpu_count=args.gpus, shared_memory=args.shared_memory),
        replicas=args.num_nodes,
        timeout=args.timeout,
    )
    if args.num_nodes > 1:
        spec.leader_selection = True
        spec.propagate_failure = True
        spec.propagate_preemption = True
    if args.no_host_networking:
        spec.host_networking = False
    else:
        spec.host_networking = True
    return spec


def main():
    args, commands = get_args()
    if args.workspace:
        beaker_client = beaker.Beaker.from_env(default_workspace=args.workspace)
    else:
        beaker_client = beaker.Beaker.from_env()
    beaker_secrets = [secret.name for secret in beaker_client.secret.list()]
    whoami = beaker_client.user.get().name
    beaker.Beaker.TIMEOUT = 300

    full_commands = [make_internal_command(command, args) for command in commands]
    for idx, full_command in enumerate(full_commands):
        console.rule(f"[bold blue]Command {idx + 1}[/bold blue]")
        console.print(Text(full_command))

    experiment_spec = beaker.BeakerExperimentSpec(
        description=args.description,
        tasks=[
            make_task_spec(args, full_command, i, beaker_secrets, whoami)
            for i, full_command in enumerate(full_commands)
        ],
        budget=args.budget,
        retry=beaker.BeakerRetrySpec(allowed_task_retries=args.max_retries),
    )

    @backoff.on_exception(backoff.expo, requests.exceptions.Timeout, max_tries=5, factor=5)
    def launch_experiment():
        exp = beaker_client.experiment.create(spec=experiment_spec)
        console.log(f"Kicked off Beaker job. https://beaker.org/ex/{exp.experiment.id}")
        return exp

    launch_experiment()


if __name__ == "__main__":
    main()
