import sys
import os
import urllib.request
import http
from goatools.obo_parser import GODag


def parse(path, out_path, go_file, new_go, godag_file):
    print('Parsing {}'.format(path))
    problem_path = 'problems_training'
    if new_go:
        problem_path += '_expanded'
    problem_path += '.txt'
    problems_file = open(problem_path, 'w')

    godag = GODag(godag_file,
                  optional_attrs=['relationship'])

    # previous_test_dataset = []
    # if os.path.isfile(out_path):
    #     with open(out_path, 'r') as ifile:
    #         for line in ifile:
    #             previous_test_dataset.append(line.split(';')[0])

    out_file = open(out_path, 'w')

    seq = ""
    sp_id = ""

    go_list = []
    with open(go_file, 'r') as ifile:
        for line in ifile:
            go_list.append(line.split()[1].split('.')[0])

    print('Found {} GO terms.'.format(len(go_list)))

    annotations = {}
    with open(path + '.txt', 'r') as ifile:
        for line in ifile:
            line = line.strip().split()
            if len(line[0]) == 0:
                continue
            if line[2] != 'F':
                continue
            if not line[1] in go_list:
                continue
            if not line[0] in annotations:
                annotations[line[0]] = [line[1]]
            else:
                annotations[line[0]].append(line[1])

    print('Found functional annotations for {} proteins.'.format(len(annotations)))

    count = 0
    problems = 0
    with open(path + '.fasta', 'r') as ifile:
        for line in ifile:
            if not line.startswith('>'):
                seq += line.strip()
                if len(seq) > 1000:
                    seq = ""
                    sp_id = ""
                    continue
            else:
                if count % 100 == 0:
                    print('Got {} proteins, of which {} annotations failed.'.format(count, problems))
                    out_file.flush()
                if not len(sp_id) == 0:
                    try:
                        if not new_go:
                            gos = annotations[sp_id]
                        else:
                            gos = retrieve_gos(sp_id)
                            if isinstance(gos, str):
                                gos = None

                        if gos:
                            parents = []
                            for go in gos:
                                try:
                                    parents += godag[go].get_all_parents()
                                except KeyError:
                                    pass
                            gos += parents
                            gos = [go for go in gos if go in go_list]
                            gos = list(set(gos))
                            if len(gos) > 0:
                                out_file.write('{};{};{}\n'.format(sp_id, seq, ','.join(gos)))
                            else:
                                raise KeyError
                        else:
                            raise KeyError

                    except KeyError:
                        problems_file.write('{}\n'.format(sp_id))
                        problems += 1

                sp_id = line.strip()[1:]
                seq = ""
                count += 1

    print('Wrote to {}'.format(out_path))

    # close files
    out_file.flush()
    problems_file.flush()
    out_file.close()
    problems_file.close()
    print('cafa training parser DONE.')


def retrieve_gos(sp_id):
    tries = 10

    gos = None
    tried = 0
    retrieved = None
    while tried < tries:
        try:
            retrieved = urllib.request.urlopen(
                'https://www.uniprot.org/uniprot/?query='
                'id:{}+AND+reviewed:yes&columns=id,reviewed,go&format=tab'.format(sp_id))
        except (http.client.RemoteDisconnected, urllib.error.HTTPError):
            tried += 1
        if retrieved:
            break

    if not retrieved:
        return sp_id

    try:
        for rt_line in retrieved:
            rt_line = rt_line.decode().strip().split('\t')
            if not rt_line[1] == 'reviewed':
                continue

            gos = [outer.split(']')[0] for outer in rt_line[2].split('[')]
            del gos[0]

    except IndexError:
        return sp_id

    return gos


if __name__ == '__main__':
    path = sys.argv[1]
    out_path = sys.argv[2]
    go_file = sys.argv[3]
    new_go = sys.argv[4].lower() in ['yes', 'y', 'true', 't']
    godag_file = sys.argv[5]

    parse(path, out_path, go_file, new_go, godag_file)
