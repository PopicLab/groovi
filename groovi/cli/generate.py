# Copyright (c) 2024-2026 Popic Lab
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU Affero General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU Affero General Public License for more details.

#You should have received a copy of the GNU Affero General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

import groovi
from groovi.cli.config import GenerateConfig
from groovi.graph.datasets import StreamingGraphDataset
import torch

torch.set_num_threads(1)

def main(config_yaml):
    print("*********************************")
    print("*  groovi (%s): graph generation   *" % groovi.__version__)
    print("*********************************")
    config = GenerateConfig(config_yaml)
    dataset = StreamingGraphDataset(config)
    for graph in dataset:
        pass
