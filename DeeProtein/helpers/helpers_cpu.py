
def count_children(GODag, go):
    return len(GODag.query_term(go).get_all_children())


def extend_gos_by_parents(GODag, gos, logger=None):
    gos_out = set()
    for go in gos:
        gos_out.add(go)
        try:
            gos_out.update(GODag.query_term(go).get_all_parents())
        except:
            if logger:
                logger.debug('Could not get parents for term {}.'.format(go))
    return gos_out


def filter_seq(seq):

    for c in seq:
        if not c in 'ACDEFGHIKLMNPQRSTVWY':
            return False
    return True


def filter_gos(gos):
    return True


def calc_secondary(secondary):
    ret = {x: 0 for x in 'HBEGITS.'} # see http://www.rcsb.org/pages/help/ssHelp
    for c in secondary:
        ret[c] += 1
    return ret


def calc_disorder(disorder):
    ret = {x: 0 for x in 'X-'}
    for c in disorder:
        ret[c] += 1
    return ret


def calc_sequence(sequence):
    ret = {x: 0 for x in 'ACDEFGHIKLMNPQRSTVWY'}
    for c in sequence:
        ret[c] += 1
    return ret