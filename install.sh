#!/bin/sh

# Install git-lfs for OSX
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
if [[ "$OSTYPE" =~ ^darwin ]]; then
  if hash git lfs 2>/dev/null; then
        git lfs install
  else
    if command -v brew; then
        brew install git-lfs
        git lfs install
    else
        echo "Please install brew and run the install script again"
    fi
  fi
fi

cd overcooked_ai
pip install -e .
cd ..

pip install -e .

conda install protobuf -y
pip install tensorflow-gpu==2.4.1
bash /human_aware_rl/run_tests.sh