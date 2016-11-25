import sys

def get_data(infile):
    infile = sys.argv[1]

    ifobj = open(infile, "r")
    ofobj = open(infile + ".processed", "w")

    lines = ifobj.readlines()[:10000]

    users = []
    items = []

    for line in lines:
        line = line.strip().split(",")
        user,item = line[0],line[1]
        users.append(user)
        items.append(item)

    print("Read file")

    users = list(set(users))
    items = list(set(items))

    ofobj.write(str(len(users)) + " " + str(len(items)))
    ofobj.write("\n")

    for user in users:
        ofobj.write(user + "\n")

    for item in items:
        ofobj.write(item + "\n")

    print("Wrote initial file")

    data = [[] for i in range(len(users))]

    prev = 0
    count = 0

    for line in lines:
        line = line.strip().split(",")
        user,item,rating = line[0],line[1],line[2]
        i = users.index(user)
        j = items.index(item)
        data[i].append((j, float(rating)))
        if int(count / len(lines)) > prev:
            prev = int(count / len(lines))
            print(prev, "% done")
        count = count + 1

    final = []

    for line in data:
        final.append(line)

    print("Read lines")

    print("Done")

    ifobj.close()
    ofobj.close()

    return final
