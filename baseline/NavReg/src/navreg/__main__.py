"""NavReg commands: serialization, one GPU index, or one full experiment."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

from .config import DEFAULT_DATA_ROOT, PROJECT_ROOT, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Serialize the complete corpus')
    prepare.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    prepare.add_argument('--output-dir', type=Path, required=True)
    prepare.add_argument('--workers', type=int, default=4)
    build = commands.add_parser('build-index', help='Chunk, embed and build one GPU Faiss index')
    run = commands.add_parser('run', help='Run one experiment; one command does not sweep K or repeat seeds')
    for command in (build, run):
        command.add_argument('--method', choices=['NavReg'], default='NavReg')
        command.add_argument('--profile', default='local', help='local, cluster, or a machine YAML path')
        command.add_argument('--experiment', type=Path)
        command.add_argument('--embedding', choices=['large', 'm3'])
        command.add_argument('--serialized-dir', type=Path)
    build.add_argument('--output-dir', type=Path, help='New bundle directory; defaults to indexes_root/embedding')
    run.add_argument('--index-dir', type=Path)
    run.add_argument('--run-dir', type=Path)
    run.add_argument('--seed', type=int)
    run.add_argument('--backend', choices=['huggingface', 'openai_api'])
    run.add_argument('--mode', choices=['non_thinking', 'thinking'])
    run.add_argument('--debug', action='store_true')
    run.add_argument('--limit', type=int)
    run.add_argument('--resume', action='store_true')
    run.add_argument('--skip-judge', action='store_true', help='Leave uncertain answer pairs pending for later evaluation')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            from .prepare import serialize_repository
            report = serialize_repository(args.data_root, args.output_dir, args.workers)
        else:
            config = load_config(args.profile, args.experiment, args.embedding, getattr(args, 'seed', None))
            if args.command == 'build-index':
                from .index import build_index
                report = build_index(config, args.output_dir, args.serialized_dir)
            else:
                if args.backend:
                    config['generation']['backend'] = args.backend
                if args.mode:
                    config['generation']['huggingface']['profile'] = args.mode
                from .runner import run_experiment
                async def execute():
                    directory = await run_experiment(config, args.run_dir, args.index_dir, args.serialized_dir,
                                                     args.debug, args.limit, args.resume)
                    sys.path.insert(0, str(PROJECT_ROOT / 'evaluation'))
                    from run import evaluate_run
                    metrics = await evaluate_run(directory, not args.skip_judge)
                    return {'run_dir': str(directory), 'evaluation': metrics}
                report = asyncio.run(execute())
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3 if report.get('evaluation', {}).get('status') == 'needs_adjudication' else 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print('NavReg cannot continue: ' + str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Interrupted. Resume the same run with --resume.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
