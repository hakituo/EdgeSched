#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/leslie/mvp_core/cpp_scheduler/build')
import os
os.environ['LD_LIBRARY_PATH'] = '/home/leslie/mvp_core/cpp_scheduler/build/bin:/home/leslie/mvp_core/cpp_scheduler/build/llama.cpp/src:' + os.environ.get('LD_LIBRARY_PATH', '')
import scheduler_py as sp
attrs = [x for x in dir(sp) if not x.startswith('_')]
for a in attrs:
    print(a)
