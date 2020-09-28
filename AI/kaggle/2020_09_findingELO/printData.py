import matplotlib.pyplot as plt

# print data point as 2d or 3d space
# data should be PCA-ed data
# n_cols : number of columns (not include target)
# df_pca : dataFrame
# title  : title of displayed chart
def printDataAsSpace(n_cols, df_pca, title):

    # set markers
    markers = ['^', 'o']

    # plot if the value of n_cols is 2
    # https://medium.com/@john_analyst/pca-%EC%B0%A8%EC%9B%90-%EC%B6%95%EC%86%8C-%EB%9E%80-3339aed5afa1
    if n_cols == 2:

        # add each point
        for i, marker in enumerate(markers):
            x_axis_data = df_pca[df_pca['target']==i]['pca0']
            y_axis_data = df_pca[df_pca['target']==i]['pca1']
            plt.scatter(x_axis_data, y_axis_data, s=1, marker=marker, label=df_pca['target'][i])

        # set labels and show
        plt.title(title)
        plt.legend()
        plt.xlabel('pca0')
        plt.ylabel('pca1')
        plt.show()

    # plot in the space, if the value of n_cols is 3
    # https://python-graph-gallery.com/372-3d-pca-result/
    elif n_cols == 3:

        fig = plt.figure()
        fig.suptitle(title)
        
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_pca['pca0'], df_pca['pca1'], df_pca['pca2'], c=df_pca['target'], s=1)

        # set labels and show
        ax.set_xlabel('pca0')
        ax.set_ylabel('pca1')
        ax.set_zlabel('pca2')
        plt.show()

    else: print('n_cols should be 2 or 3 to print data as 2d/3d space')
