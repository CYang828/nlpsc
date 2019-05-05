# encoding:utf-8

import os

import aiofiles


def gen_filename(name, prefix=None, suffix=None):
    if prefix:
        name = '{}_{}'.format(prefix, name)

    if suffix:
        dotpos = name.rfind('.')
        if dotpos > 0:
            name = name[:dotpos+1] + suffix + name[dotpos:]
        else:
            name += '.{}'.format(suffix)
    return name


def get_files(fin, filter_func=None):
    if os.path.isdir(fin):
        filelist = []
        for root, dirs, files in os.walk(fin, topdown=False):
            for name in files:
                if filter_func and filter_func(name):
                    filelist.append((name,
                                     os.path.join(root, name)))
                else:
                    filelist.append((name,
                                     os.path.join(root, name)))
        return filelist
    elif os.path.isfile(fin):
        name = os.path.basename(fin)
        return [(name, fin)]
    else:
        print("{} it's noexist or a special file(socket,FIFO,device file), please check your input location".format(fin))
        raise OSError


def create_file(output_dir, name, text):
    if os.path.isdir(output_dir):
        print('make sure {0} exist, `mkdir -p {0}` may fix it'.format(output_dir))
        raise OSError

    path = os.path.join(output_dir, name)
    with open(path, 'w+', encoding='utf8') as f:
        f.write(text)
    return path


async def aio_read_file(path, pattern='r', encoding='utf-8'):
    async with aiofiles.open(path, mode=pattern, encoding=encoding) as f:
        contents = await f.read()
        return contents


async def aio_write_file(outdir, name, content, pattern='w', encoding='utf-8'):
    path = os.path.join(outdir, name)
    try:
        async with aiofiles.open(path, mode=pattern, encoding=encoding) as f:
            await f.write(content)
            return path
    except FileNotFoundError as e:
        print('make sure {0} exist, `mkdir -p {0}` may fix it'.format(outdir))
        return

