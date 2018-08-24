from datapreprocessing import DataPreprocessor


def main():
    # make GO_file by wc -l * > GO_file.txt
    GO_file_list = [
                    'path/to/gofile'
                    ]

    dp = DataPreprocessor(save_dir='path/to/save/dir',
                          read_dir='path/to/swissprot/downloads',
                          godagfile='')

    print('Got DataPreprocessor')
    for GO_file in GO_file_list:
        print('go file: {}'.format(GO_file))
        dp.generate_dataset_by_GO_list(GO_file, GO_cat='function')
    print('DONE')


if __name__ == '__main__':
    main()
