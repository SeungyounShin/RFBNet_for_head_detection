if __name__ == '__main__':

    f_full = open('../DATASET/HollywoodHeads/Splits/train.txt', 'r')
    f_light = open('../DATASET/HollywoodHeads/Splits/train_light.txt', 'w')

    i = 0
    while True:
        i = i + 1
        line = f_full.readline()
        if not line: break
        if i % 100 == 0:
            f_light.write(line)
            print(i)
    f_full.close()
    f_light.close()
