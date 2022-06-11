import librosa
import librosa.display
import numpy as np
import warnings
import os
import scipy.stats as st
from scipy.spatial.distance import cityblock, cosine
import statistics

# Path of the file to read (Change according to your directory)
FILE_PATH = "../resources"
# Path of the file to save (Change according to your directory)
SAVE_PATH = "../resources"


# Prints
printOriginal = False
printModified = False
printNormalized = False

# List of musics
musicsRaw = np.genfromtxt(FILE_PATH + "/top100_features.csv", delimiter=",", dtype="str")
musics = musicsRaw[1:, 0]


# Function to read top100_features.csv and generate the correct numpy matrix (2.1.1)
def top100featuresReader():
    print("Reading top100_features.csv...")
    matrix = np.genfromtxt(FILE_PATH + "/top100_features.csv", delimiter=",")

    if printOriginal:
        print("\nShape of Top 100 Matrix: {}x{}\n".format(matrix.shape[0], matrix.shape[1]))
        print(matrix)

    matrix = matrix[1:, 1:(matrix.shape[1] - 1)]

    if printModified:
        print("\nShape of Top 100 Matrix Modified: {}x{}\n".format(matrix.shape[0], matrix.shape[1]))
        print(matrix)

    print("Finished reading")
    return matrix


# Normalize Top 100 features
def normalizeFeatures(data):
    print("Normalizing features...")

    dataNormalized = np.zeros(data.shape)
    for i in range(data.shape[1]):
        dataNormalized[:, i] = (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min())

    if printNormalized:
        print("\nShape of Top 100 matrix normalized: {}x{}\n".format(dataNormalized.shape[0], dataNormalized.shape[1]))
        print(dataNormalized)

    np.savetxt(SAVE_PATH + "/top100_features_normalized.csv", dataNormalized, delimiter=",")
    print("Finished normalizing features")
    return dataNormalized


# Extract librosa features and calculate statistics (2.2.1 & 2.2.2)
def statsReader():
    if os.path.exists(SAVE_PATH + "/stats.csv"):
        print("Reading stats.csv...")
        data = np.genfromtxt(SAVE_PATH + "/stats.csv", delimiter=",")
        print("Finished reading stats")
        return data

    print("No 'stats_normalized.csv' file found. Generating stats from files in /musics...")

    directory = os.listdir(FILE_PATH + "/musics")
    f = open(SAVE_PATH + "/features.txt", "w")

    music = 0
    stats = []

    for file in sorted(directory):
        if file == ".DS_Store":
            continue

        features = []

        filey = librosa.load(FILE_PATH + "/musics/" + file, sr=22050, mono=True)[0]
        features.append(librosa.feature.mfcc(filey, hop_length=512, win_length=2048, n_mfcc=13))
        features.append(librosa.feature.spectral_centroid(filey, hop_length=512, win_length=2048))
        features.append(librosa.feature.spectral_bandwidth(filey, hop_length=512, win_length=2048))
        features.append(librosa.feature.spectral_contrast(filey, hop_length=512, win_length=2048))
        features.append(librosa.feature.spectral_flatness(filey, hop_length=512, win_length=2048))
        features.append(librosa.feature.spectral_rolloff(filey, hop_length=512, win_length=2048))
        features.append(librosa.yin(filey, hop_length=512, frame_length=2048, fmin=20, fmax=11025))
        features[len(features) - 1][features[len(features) - 1] == 11025] = 0
        features.append(librosa.feature.rms(filey, hop_length=512, frame_length=2048))
        features.append(librosa.feature.zero_crossing_rate(filey, hop_length=512, frame_length=2048))
        features.append(librosa.beat.tempo(filey, hop_length=512))

        tempStats = []

        for i in range(len(features)):
            if i == 6:
                tempStats.append(np.mean(features[i]))
                tempStats.append(np.std(features[i]))
                tempStats.append(st.skew(features[i]))
                tempStats.append(st.kurtosis(features[i]))
                tempStats.append(statistics.median(features[i]))
                tempStats.append(max(features[i]))
                tempStats.append(min(features[i]))
            elif i == 9:
                tempStats.append(features[9])
            else:
                for j in range(features[i].shape[0]):
                    tempStats.append(np.mean(features[i][j, :]))
                    tempStats.append(np.std(features[i][j, :]))
                    tempStats.append(st.skew(features[i][j, :]))
                    tempStats.append(st.kurtosis(features[i][j, :]))
                    tempStats.append(statistics.median(features[i][j, :]))
                    tempStats.append(max(features[i][j, :]))
                    tempStats.append(min(features[i][j, :]))

        stats.append(tempStats)
        music += 1

        if os.path.exists(FILE_PATH + "/musics/.DS_Store"):
            print("Finished music {}/{}".format(music, len(directory) - 1))
        else:
            print("Finished music {}/{}".format(music, len(directory)))
    f.close()

    np.savetxt(SAVE_PATH + "/stats.csv", stats, delimiter=",")
    print("Finished generating stats")
    return np.array(stats)


# Function responsible for normalization of the statistics (2.1.2 & 2.2.3)
def normalizeStats(data):
    if os.path.exists(SAVE_PATH + "/stats_normalized.csv"):
        print("Reading stats_normalized.csv...")
        dataNormalized = np.genfromtxt(SAVE_PATH + "/stats_normalized.csv", delimiter=",")
        print("Finished reading stats")
        return dataNormalized

    print("Normalizing statistics...")

    dataNormalized = np.zeros(data.shape)
    for i in range(data.shape[1]):
        dataNormalized[:, i] = (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min())

    if printNormalized:
        print("\nShape of statistics matrix normalized: {}x{}\n".format(dataNormalized.shape[0], dataNormalized.shape[1]))
        print(dataNormalized)

    np.savetxt(SAVE_PATH + "/stats_normalized.csv", data, delimiter=",")

    print("Finished normalizing statistics")
    return dataNormalized


# Calculate the distance for all 6 matrix (top100 and statistics, both normalized) (3.1)
def distCalculus(matrix, features):
    print("Calculating the distances...")

    euclidian = np.zeros((matrix.shape[0], matrix.shape[0]))
    manhattan = np.zeros((matrix.shape[0], matrix.shape[0]))
    coseno = np.zeros((matrix.shape[0], matrix.shape[0]))
    euclidian_top100 = np.zeros((features.shape[0], features.shape[0]))
    manhattan_top100 = np.zeros((features.shape[0], features.shape[0]))
    coseno_top100 = np.zeros((features.shape[0], features.shape[0]))
    matrix = np.delete(matrix, 174, 1)

    for i in range(len(matrix)):
        for j in range(i+1, len(matrix)):
            euclidian[i][j] = np.linalg.norm((matrix[i] - matrix[j]))
            euclidian[j][i] = np.linalg.norm((matrix[i] - matrix[j]))
            manhattan[i][j] = cityblock(matrix[i], matrix[j])
            manhattan[j][i] = cityblock(matrix[i], matrix[j])
            coseno[i][j] = cosine(matrix[i], matrix[j])
            coseno[j][i] = cosine(matrix[i], matrix[j])

            euclidian_top100[i][j] = np.linalg.norm((features[i] - features[j]))
            euclidian_top100[j][i] = np.linalg.norm((features[i] - features[j]))
            manhattan_top100[i][j] = cityblock(features[i], features[j])
            manhattan_top100[j][i] = cityblock(features[i], features[j])
            coseno_top100[i][j] = cosine(features[i], features[j])
            coseno_top100[j][i] = cosine(features[i], features[j])

    np.savetxt(SAVE_PATH + "/euclidian.csv", euclidian, delimiter=",")
    np.savetxt(SAVE_PATH + "/manhattan.csv", manhattan, delimiter=",")
    np.savetxt(SAVE_PATH + "/coseno.csv", coseno, delimiter=",")
    np.savetxt(SAVE_PATH + "/euclidian_top100.csv", euclidian_top100, delimiter=",")
    np.savetxt(SAVE_PATH + "/manhattan_top100.csv", manhattan_top100, delimiter=",")
    np.savetxt(SAVE_PATH + "/coseno_top100.csv", coseno_top100, delimiter=",")

    print("Finished calculating distances")
    return [euclidian, manhattan, coseno, euclidian_top100, manhattan_top100, coseno_top100]


# Generate the rankings of a chosen matrix (top 20 musics with the lowest distance) (3.3)
def ranking(matrix, n):
    # 10 / 26 / 28 / 59
    ranks = np.zeros((4, 20))

    ranks[0] = ((matrix[10]).argsort()[:n])[1:]
    ranks[1] = ((matrix[26]).argsort()[:n])[1:]
    ranks[2] = ((matrix[28]).argsort()[:n])[1:]
    ranks[3] = ((matrix[59]).argsort()[:n])[1:]

    for i in range(4):
        print("Query ", i + 1, ":")
        for j in range(20):
            if j < 9:
                print(" -> Music ", j + 1, "", musics[int(ranks[i][j])])
            else:
                print(" -> Music ", j + 1, musics[int(ranks[i][j])])
        print("")

    return ranks


# Extract metadata for all musics and give points to the similarities between them and the query (4.1.1)
def metadataExtraction():
    metadataDefault = np.genfromtxt(FILE_PATH + "/panda_dataset_taffc_metadata.csv", delimiter=",", dtype="str")
    metadata = metadataDefault[1:, [1, 3, 9, 11]]
    quality = np.zeros((4, 900))
    value = 0
    present = [10, 26, 28, 59]

    for k in range(4):
        for i in range(metadata.shape[0]):
            if i != present[k]:
                for j in range(4):
                    if j > 1:
                        listA = metadata[present[k], j][1:-1].split('; ')
                        listB = metadata[i, j][1:-1].split('; ')
                        for genre in listA:
                            for genre2 in listB:
                                if genre == genre2:
                                    value = value + 1
                    else:
                        if metadata[present[k], j] == metadata[i][j]:
                            value += 1
                quality[k][i] = value
                value = 0
            else:
                quality[k][i] = -1

    np.savetxt(SAVE_PATH + "/similarities.csv", quality, delimiter=",")
    return quality


# Create a ranking based on it for the 4 musics (4.1.2)
def metadataRanking(matrix, n):
    ranks = np.zeros((4, 20))

    ranks[0] = (-matrix[0]).argsort()[:n]
    ranks[1] = (-matrix[1]).argsort()[:n]
    ranks[2] = (-matrix[2]).argsort()[:n]
    ranks[3] = (-matrix[3]).argsort()[:n]

    print("\nMetadata rankings:\n")
    for i in range(4):
        print("Query ", i+1, ":")
        for j in range(20):
            if j < 9:
                print(" -> Music ", j + 1, "", musics[int(ranks[i][j])])
            else:
                print(" -> Music ", j + 1, musics[int(ranks[i][j])])
        print("")

    return ranks


def precision(m1, m2):
    for i in range(4):
        print("Precision for query ", i, ": ", len(np.intersect1d(m1[i], m2[i])) / 20)


# Main Function
def main():
    warnings.filterwarnings("ignore")

    # Features
    features = top100featuresReader()
    featuresNormalized = normalizeFeatures(features)

    # Stats
    stats = statsReader()
    statsNormalized = normalizeStats(stats)

    quality = metadataExtraction()
    ranks2 = metadataRanking(quality, 20)

    distances = distCalculus(statsNormalized, featuresNormalized)
    functions = ["Euclidian", "Manhattan", "Coseno", "Euclidian Top 100", "Manhattan Top 100", "Coseno top 100"]
    for i in range(len(distances)):
        print("\nDistance Ranking (", functions[i], "):\n")
        ranks = ranking(distances[i], 21)
        precision(ranks, ranks2)


# Main Function
if __name__ == '__main__':
    main()