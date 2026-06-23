"""Compatibility entry for plot_markov_results.py.

如果运行时少输入了文件名开头的 p，也可以通过这个脚本转到真正的
绘图脚本。主要逻辑仍在 plot_markov_results.py 中维护。
"""

from __future__ import annotations

from plot_markov_results import main


if __name__ == "__main__":
    main()
