import sys
import numpy as np
import random
import argparse

def segment_matrix(array,max_num, label_num):
    '''
    Labels the matrix into different labels (according to thresholds created according to the label number)
    :param array: the input array, our data that we are working on
    :param max_num: the highest intesnity pixel
    :param label_num: number of desired labels
    :return: the index array, which will have the same size as array, but will contain at each coordinate the label associated to that coordinate
    '''
    label_interval = int(max_num/label_num)
    index_array = np.zeros(array.shape)
    for i in range(label_num):
        lower_bound = np.full_like(array,(max_num - (i+1)*label_interval))
        upper_bound = np.full_like(array,(max_num - i*label_interval))
        mask = (array > lower_bound) & (array <= upper_bound)
        mask.astype(np.float)
        index_array += mask*(i+1)
    return index_array


def create_obj(verticies, faces, object_name, number_of_classes):
    '''
    Outputs the veriticis and faces into object files, it will create a separate obj. file for each bin, that contains verticies and faces (the on that does not will be ignored)
    it also outputs a mtl file containing same number of randomly genereated color materials as the existing classes
    :param verticies: contains class amount of lists, each containing the verticies corresponding to gives class
    :param faces: contains class amount of lists, each containing the faces corresponding to gives class
    :param object_name: a simple string, containing the name of the output objects
    :param number_of_classes: the amount of desired classes
    '''
    counter = 0
    # Iterate over every class
    for j in range(number_of_classes):
        h = len(verticies[j])
        # Check if they contain any verticies
        if h > 0:
            h_f = len(faces[j])
            f = open( object_name + str(j) +'.obj','w')
            # Put the color coding
            f.write('mtllib' + ' ' + 'materials.mtl' + '\n')
            f.write('usemtl' + ' ' + 'mat' + str(counter) + '\n')
            # Output the verticies for a given class
            for i in range(h):
                f.write('v ' + str(verticies[j][i][0]) + ' ' + str(verticies[j][i][1]) + ' ' + str(verticies[j][i][2]) + '\n')
            # Output the faces for a gives class
            for i in range(h_f):
                f.write('f ' + str(int(faces[j][i][0])) + ' ' + str(int(faces[j][i][1])) + ' ' + str(int(faces[j][i][2])) + '\n')
            f.close()
            counter += 1

    f = open('mymaterials'+'.mtl','w')
    for j in range(counter):
        R = random.random()
        G = random.random()
        B = random.random()
        f.write('newmtl'+' '+ 'mat'+ str(j)+'\n')
        f.write('    '+ 'Kd'+ ' ' + str(R) + ' ' +  str(G) + ' ' + str(B)+'\n')

def addFace(i, j, k, vertices, faces, coord):
    '''

    :param i: i coordinate of current voxel
    :param j: j coordinate of current voxel
    :param k: k coordinate of current voxel
    :param vertices: the corresponding vertices list of current pixel label
    :param faces: the corresponding faces list of current pixel label
    :param coord: the different label is detected from which coordinate direction
    :return: modified vertices and faces lists
    '''
    faces_length = len(vertices) + 1
    if coord == 1:
        vertices.append([i, j + 0.5, k + 0.5])
        vertices.append([i, j + 0.5, k - 0.5])
        vertices.append([i, j - 0.5, k + 0.5])
        vertices.append([i, j - 0.5, k - 0.5])
    elif coord == 2:
        vertices.append([i + 0.5, j, k + 0.5])
        vertices.append([i + 0.5, j, k - 0.5])
        vertices.append([i - 0.5, j, k + 0.5])
        vertices.append([i - 0.5, j, k - 0.5])

    elif coord == 3:
        vertices.append([i + 0.5, j + 0.5, k])
        vertices.append([i + 0.5, j - 0.5, k])
        vertices.append([i - 0.5, j + 0.5, k])
        vertices.append([i - 0.5, j - 0.5, k])

    faces.append([faces_length, faces_length + 1, faces_length + 2])
    faces.append([faces_length + 1, faces_length + 3, faces_length + 2])

def iterateVoxels(array, index_array, number_of_labels):
    '''

    :param array: the data array with shape (64,64,64)
    :param index_array: the label array with same shape (64,64,64)
    :param number_of_labels: the desired number of labels from user input
    :return: the genereated vertices and faces lists where the labels of neighboring voxels change
    the number of vertices/faces lists are the same as the desired number_of_labels, but some of them will be empty
    '''
    vertices = [[] for i in range(number_of_labels)]
    faces = [[] for i in range(number_of_labels)]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            for k in range(array.shape[2]):
                if index_array[i, j, k] == 0:
                    continue
                if i > 0:
                    if (index_array[i - 1, j, k] != index_array[i, j, k]):
                        addFace(i - 0.5, j, k, vertices[int(index_array[i, j, k])-1], faces[int(index_array[i, j, k])-1], 1)
                if i < array.shape[0] - 2:
                    if (index_array[i + 1, j, k] != index_array[i, j, k]):
                        addFace(i + 0.5, j, k, vertices[int(index_array[i, j, k])-1], faces[int(index_array[i, j, k])-1], 1)
                if j > 0:
                    if (index_array[i, j - 1, k] != index_array[i, j, k]):
                        addFace(i, j - 0.5, k, vertices[int(index_array[i, j, k])-1], faces[int(index_array[i, j, k])-1], 2)
                if j < array.shape[1] - 2:
                    if (index_array[i, j + 1, k] != index_array[i, j, k]):
                        addFace(i, j + 0.5, k, vertices[int(index_array[i, j, k])-1], faces[int(index_array[i, j, k])-1], 2)
                if k > 0:
                    if (index_array[i, j, k - 1] != index_array[i, j, k]):
                        addFace(i, j, k - 0.5, vertices[int(index_array[i, j, k])-1], faces[int(index_array[i, j, k])-1], 3)
                if k < array.shape[2] - 2:
                    if (index_array[i, j, k + 1] != index_array[i, j, k]):
                        addFace(i, j, k + 0.5, vertices[int(index_array[i, j, k])-1], faces[int(index_array[i, j, k])-1], 3)

    return vertices, faces

def readData(data_name):
    '''

    :param data_name: the raw pgm3d file
    :return: the reshaped 3d array and the max number of the dataset
    '''
    try:
        f = open(data_name, "r")
        dataset = f.read()

        dataset = dataset.splitlines()
        data_format = dataset[0]
        x, y, z = dataset[1].split()
        max_num = int(dataset[2])
        # Step 3. Put the data values in a 3D numpy array.

        array_3d = np.array(dataset[3:], dtype=int)
        array_3d = array_3d.reshape(int(x), int(y), int(z))
        return array_3d, max_num

    except IOError:
        print('Please add a valid filename')
        exit()


def usage():
    # Prints the instructions if there is no user input
    print('Wrong arguments, please add a pgm3d file and the number of labels (between 1 and 255)')

if __name__ == "__main__":
        '''
    parser = argparse.ArgumentParser(description='Please enter the pgm3d file and number of classes you want:')
    parser.add_argument('filename', type=str,
                        help='a pgm3d file name')
    parser.add_argument('--number',  type=int, default=5,
                        help='the number of classes of your data')
    args = parser.parse_args()
    '''
    if len(sys.argv)  < 3:
        usage()
    else:
        try:
            number_of_labels = int(sys.argv[2])
            if int(number_of_labels) > 255 or int(number_of_labels) < 0:
                usage()
                exit()
            array = sys.argv[0]
            array_3d, max_num = readData(sys.argv[1])
            indexmap = segment_matrix(array_3d, max_num, number_of_labels)
            verticies, faces = iterateVoxels(array_3d, indexmap, number_of_labels)
            create_obj(verticies, faces, 'object', number_of_labels)
        except ValueError:
            print('Please enter an integer as label numbers')