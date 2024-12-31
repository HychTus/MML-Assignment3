# 将当前目录加入环境变量
# 使得 python import 时能够 import 高层的 module
export PYTHONPATH=$PYTHONPATH:/data/chy/others/MML-Assignment3


alias gitlog="git log --oneline --graph --all"
export GIT_SSH_COMMAND="ssh -i /data/chy/private_key"
export TMPDIR="/data/chy/tmp"
# export PIP_CACHE_DIR="data/cache/pip"
# 之前写错了, 会导致 pip 保存缓存到创建环境时的目录下
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH}:/data/chy/envs"


# ====== 启用补全 ======
if [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
fi

# ====== 颜色支持 ======
alias ls='ls --color=auto'
alias grep='grep --color=auto'

# ====== 自定义提示符 (PS1) ======
RESET='\[\033[0m\]'
GREEN='\[\033[0;32m\]'
BLUE='\[\033[0;34m\]'
PS1="${GREEN}\u@\h${RESET}:${BLUE}\w${RESET}\$ "


tmux source-file /data/chy/.tmux.conf


# ====== 加载成功提示 ======
# echo -e "${GREEN}Extra bashrc activated.${RESET}"
clear