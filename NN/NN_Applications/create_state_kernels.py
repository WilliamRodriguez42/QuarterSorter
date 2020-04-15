from id_stor.manage_id_stor import read_id_stor, create_id_stor
import lib.coin_center as cc
import numpy as np
import matplotlib.pyplot as plt

states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
def create_state_kernels(j_pool):

    id_stor = create_id_stor(cc.dataset.shape[0])
    read_id_stor(id_stor)

    side_width = np.zeros(26)
    count = np.zeros(26)

    for coin_ids in id_stor:
        for ids in coin_ids:

            if len(ids) > 0:
                ((px, _), pc, _) = ids[0]

                for ((x, _), c, _) in ids[1:]:
                    ci = ord(c) - ord('a')

                    dx = x - px

                    side_width[ci] += dx / 2
                    count[ci] += 1
                    px = x


    space_side_width = 20


    nanners = count == 0
    count[nanners] = 1
    side_width /= count

    mean_side_width = side_width[np.logical_not(nanners)].mean()
    side_width[nanners] = mean_side_width
    print(mean_side_width)

    max_sw = int(side_width.max()+0.5)
    longest_name = 0
    for state in states:
        if len(state) > longest_name:
            longest_name = len(state)

    state_kernels = np.zeros((50, longest_name * max_sw * 2, 26))
    max_x = 0
    for i, state in enumerate(states):
        state = state.lower()

        x = 0
        for c in state:
            if c == ' ':
                x += space_side_width*2
            else:
                ci = ord(c) - ord('a')

                sw = side_width[ci]
                swi = int(sw + 0.5)

                x += sw

                xi = int(x + 0.5)
                state_kernels[i, xi-swi:xi+swi, ci] = 1

                x += sw

                if x > max_x:
                    max_x = x
    state_kernels = state_kernels[:, :int(max_x)+1, :]

    max_pool_count = np.zeros((50, state_kernels.shape[1] // j_pool + 1, 26))
    for j in range(state_kernels.shape[1]):
        v = state_kernels[:, j]

        mj = j // j_pool

        max_pool_count[:, mj] += v

    prev_max_pool_count_shape = max_pool_count.shape
    max_pool_count = max_pool_count.reshape(50, -1).T / np.sqrt(np.sum(np.square(max_pool_count), axis=(1, 2)))
    max_pool_count = max_pool_count.T.reshape(prev_max_pool_count_shape)

    #utah_index = states.index('Utah')
    #iowa_index = states.index('Iowa')
    #max_pool_count[utah_index] *= 1.1
    #max_pool_count[iowa_index] *= 0.9

    return max_pool_count

if __name__ == "__main__":
    state_kernels = create_state_kernels(6)
    print(state_kernels.shape)
    k = state_kernels[28].T

    plt.imshow(-k, cmap="Greys")
    plt.show()
