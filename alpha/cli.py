from __future__ import annotations

import argparse

from alpha.pipeline import format_report, load_config, run_research


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="alpha", description="横截面多因子研究")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="按配置文件跑完整研究流程")
    r.add_argument("config")
    r.add_argument("--out", default=None, help="输出目录（覆盖配置里的 output_dir）")
    sub.add_parser("factors", help="列出已注册的因子")
    args = p.parse_args(argv)

    if args.cmd == "factors":
        from alpha.factors import FACTOR_REGISTRY

        for name, cls in sorted(FACTOR_REGISTRY.items()):
            doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
            print(f"{name:<16}{doc}")
        return

    cfg = load_config(args.config)
    res = run_research(cfg, out_dir=args.out)
    print(format_report(res))


if __name__ == "__main__":
    main()
