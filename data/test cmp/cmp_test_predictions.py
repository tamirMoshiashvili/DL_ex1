
if __name__ == '__main__':
    """
    Compare the files: "his_prediction" and "test.pred".
    Create a new file "cmp.info" which will contain only diffs in the two files,
    every line is in the next format:
        <line-number><TAB><his-prediction><TAB><my-prediction>
    """

    # extract predictions
    his_data = [x.strip() for x in open('his_prediction', 'r').readlines()]
    my_data = [x.strip() for x in open('test.pred', 'r').readlines()]

    # compare and write to files
    cmp_file = open('cmp.info', 'w')
    for i, pair in enumerate(zip(his_data, my_data)):
        his = pair[0]
        my = pair[1]
        if his != my:
            cmp_file.write(str(i+1) + " " + his + " " + my + "\n")

    cmp_file.close()
