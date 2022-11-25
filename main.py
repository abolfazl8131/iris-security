from vision import IrisDetection

def main():
        
    b = IrisDetection()

    b.run()

    print(b.return_iris())


if __name__ == '__main__':
    try:
        main()
    except:
        pass