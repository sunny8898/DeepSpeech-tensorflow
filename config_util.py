#encoding:utf-8

from collections import defaultdict


def get_config():
    config_dict = defaultdict(str)
    with open("config",'r') as f:
        for line in f:
            if '=' not in line:
                continue
            if line.startswith('#'):
                continue
            key, value = line.split('=')
            config_dict[key] = value.strip('\n')
    return config_dict


def get_value(key):
    config_dict = get_config()
    return config_dict[key]


if __name__ == '__main__':
    print get_value('a')
