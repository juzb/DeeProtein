import os

def read_cas_domain_insertion(path):
    """
    Read the dms dataset for cas9.
    """
    # Profiling of engineering hotspots identifies an allosteric CRISPR-Cas9 switch
    filename = 'spcas9_domain_insertion_with_log'
    pdbid = '4UN3'
    chain = 'B'
    fpath = os.path.join(path, 'raw/{}.csv'.format(filename))
    problems = 0

    wt_seq = 'GAASMDKKYSIGLDIGTNSVGWAVITDEYKVPSKKFKVLGNTDRHSIKKNLIGALLFDSGETAEATRLKRTARRRYTRRKNR' \
             'ICYLQEIFSNEMAKVDDSFFHRLEESFLVEEDKKHERHPIFGNIVDEVAYHEKYPTIYHLRKKLVDSTDKADLRLIYLALAH' \
             'MIKFRGHFLIEGDLNPDNSDVDKLFIQLVQTYNQLFEENPINASGVDAKAILSARLSKSRRLENLIAQLPGEKKNGLFGNLI' \
             'ALSLGLTPNFKSNFDLAEDAKLQLSKDTYDDDLDNLLAQIGDQYADLFLAAKNLSDAILLSDILRVNTEITKAPLSASMIKR' \
             'YDEHHQDLTLLKALVRQQLPEKYKEIFFDQSKNGYAGYIDGGASQEEFYKFIKPILEKMDGTEELLVKLNREDLLRKQRTFD' \
             'NGSIPHQIHLGELHAILRRQEDFYPFLKDNREKIEKILTFRIPYYVGPLARGNSRFAWMTRKSEETITPWNFEEVVDKGASA' \
             'QSFIERMTNFDKNLPNEKVLPKHSLLYEYFTVYNELTKVKYVTEGMRKPAFLSGEQKKAIVDLLFKTNRKVTVKQLKEDYFK' \
             'KIECFDSVEISGVEDRFNASLGTYHDLLKIIKDKDFLDNEENEDILEDIVLTLTLFEDREMIEERLKTYAHLFDDKVMKQLK' \
             'RRRYTGWGRLSRKLINGIRDKQSGKTILDFLKSDGFANRNFMQLIHDDSLTFKEDIQKAQVSGQGDSLHEHIANLAGSPAIK' \
             'KGILQTVKVVDELVKVMGRHKPENIVIEMARENQTTQKGQKNSRERMKRIEEGIKELGSQILKEHPVENTQLQNEKLYLYYL' \
             'QNGRDMYVDQELDINRLSDYDVDAIVPQSFLKDDSIDNKVLTRSDKNRGKSDNVPSEEVVKKMKNYWRQLLNAKLITQRKFD' \
             'NLTKAERGGLSELDKAGFIKRQLVETRQITKHVAQILDSRMNTKYDENDKLIREVKVITLKSKLVSDFRKDFQFYKVREINN' \
             'YHHAHDAYLNAVVGTALIKKYPKLESEFVYGDYKVYDVRKMIAKSEQEIGKATAKYFFYSNIMNFFKTEITLANGEIRKRPL' \
             'IETNGETGEIVWDKGRDFATVRKVLSMPQVNIVKKTEVQTGGFSKESILPKRNSDKLIARKKDWDPKKYGGFDSPTVAYSVL' \
             'VVAKVEKGKSKKLKSVKELLGITIMERSSFEKNPIDFLEAKGYKEVKKDLIIKLPKYSLFELENGRKRMLASAGELQKGNEL' \
             'ALPSKYVNFLYLASHYEKLKGSPEDNEQKQLFVEQHKHYLDEIIEQISEFSKRVILADANLDKVLSAYNKHRDKPIREQAEN' \
             'IIHLFTLTNLGAPAAFKYFDTTIDRKRYTSTKEVLDATLIHQSITGLYETRIDLSQLGGD'

    # from ss_dis_file
    with open(fpath, mode='r', encoding='utf-8') as ifile:
        data = [[[['0', wt_seq[0]]], ['0', '0', '0', '0']]]  # [[pos1, aa1], ...], [score1, score2]
        ifile.readline()

        field_names = ifile.readline().strip().split(';')[1:]

        for line in ifile:
            fields = line.strip().split(';')
            pos = int(fields[0])

            data.append([[[pos, 'X']], [fields[x+1].replace(',', '.') for x in range(4)]])

        print("Done with {}. Got {} problems".format(filename, problems))
        filename = '{}_{}_{}'.format(pdbid, chain, filename)
        return data, wt_seq, field_names, filename


def read_mapk_1(path):
    """
    Read the dms dataset for the MAPK1/ERK2 Missense Mutants.
    """
    # Phenotypic Characterization of a Comprehensive Set of MAPK1/ERK2 Missense Mutants
    filename = 'mapk_1'
    pdbids = ['4QTE', '4QTA', '4FMQ']
    chains =  ['A',    'A',     'A']
    fpath = os.path.join(path, 'raw/{}.csv'.format(filename))
    problems = 0

    wt_seq = 'SMAAAAAAGAGPEMVRGQVFDVGPRYTNLSYIGEGAYGMVCSAYDNVNKVRVAIKKISPFEHQTYCQRTLREIKILLRFRHE' \
             'NIIGINDIIRAPTIEQMKDVYIVQDLMETDLYKLLKTQHLSNDHICYFLYQILRGLKYIHSANVLHRDLKPSNLLLNTTCDL' \
             'KICDFGLARVADPDHDHTGFLTEYVATRWYRAPEIMLNSKGYTKSIDIWSVGCILAEMLSNRPIFPGKHYLDQLNHILGILG' \
             'SPSQEDLNCIINLKARNYLLSLPHKNKVPWNRLFPNADSKALDLLDKMLTFNPHKRIEVEQALAHPYLEQYYDPSDEPIAEA' \
             'PFKFDMELDDLPKEKLKELIFEETARFQPGYRS'

    with open(fpath, mode='r', encoding='utf-8') as ifile:
        data = [[[['0', wt_seq[0]]], ['0.0', '0.0', '0.0', '0.0']]]  # [[pos1, aa1], ...], [score1, score2]

        header = ifile.readline().strip().split(';')

        field_names = ['ETP_AVERAGE', 'DOX_Average', 'SCH_Average', 'VRT_AVERAGE']

        for line in ifile:
            fields = line.strip().split(';')
            pos = int(fields[header.index('ERK2_Residue')])
            mut = fields[header.index('Mutant_AA')]
            scores = [fields[header.index(col)].replace(',', '.') for col in field_names]
            data.append([[[pos, mut]], scores])

        field_names = [f.lower() for f in field_names]

        print("Done with {}. Got {} problems".format(filename, problems))
        filenames = ['{}_{}_{}'.format(pdbid, chain, filename) for (pdbid, chain) in zip(pdbids, chains)]

        return data, wt_seq, field_names, filenames


def write_dms_set(path, data, wt_seq, field_names, filename):
    """
    Wite dms set for specified parameters.
    """
    opath = os.path.join(path, 'datasets/{}.txt'.format(filename))
    with open(opath, 'w') as ofile:
        ofile.write('seqID\t{}\n'.format('\t'.join(field_names)))
        ofile.write(wt_seq + '\n')
        for line in data:
            line_str = ''

            pos_aa_info = []
            for pos_aa in line[0]:
                pos_aa_info.append('{}-{}'.format(pos_aa[0], pos_aa[1]))
            line_str += ','.join(pos_aa_info)
            line_str += '\t{}\n'.format('\t'.join(line[1]))

            ofile.write(line_str)


path = '.'
print('Starting in {}'.format(path))


data, wt_seq, field_names, filename = read_mapk_1(path)
write_dms_set(path, data, wt_seq, field_names, filename)


data, wt_seq, field_names, filename = read_cas_domain_insertion(path)
write_dms_set(path, data, wt_seq, field_names, filename)


