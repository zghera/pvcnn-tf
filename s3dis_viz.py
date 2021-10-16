"""Demo to visualize data pipeline output."""
import matplotlib.pyplot as plt
import tensorflow as tf
from dataloaders.s3dis import create_s3dis_dataset

objects = {
    0: 'clutter',
    1: 'ceiling',
    2: 'floor',
    3: 'wall',
    4: 'beam',
    5: 'column',
    6: 'door',
    7: 'window',
    8: 'table',
    9: 'chair',
    10: 'sofa',
    11: 'bookcase',
    12: 'board'
}
dataset = create_s3dis_dataset('./data/s3dis/pointcnn/', shuffle_size=10,
    batch_size=1, num_points=7500, use_normalized_coords=False, holdout_area=5,
    is_train_split=True)
x, y = tuple(tf.squeeze(tensor) for tensor in next(iter(dataset)))
x = x[:3,:]
print(f'sample shape = {x.shape} | label shape = {y.shape}')
# for i in range(y.shape[0]):
#     print((x[:,i].numpy(), y[i]))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(y.shape[0]):
  # if int(y[i]) == 3:
    # ax.scatter(x[0,i], x[1,i], x[2,i])
  if int(y[i]) > 3:
    ax.scatter(x[0,i], x[1,i], x[2,i], label=objects[int(y[i])])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# handles, labels = ax.get_legend_handles_labels()
# handle_list, label_list = [], []
# for handle, label in zip(handles, labels):
#   if label not in label_list:
#     handle_list.append(handle)
#     label_list.append(label)
# plt.legend(handle_list, label_list)

def legend_without_duplicate_labels(ax):
  '''https://stackoverflow.com/a/56253636'''
  handles, labels = ax.get_legend_handles_labels()
  unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
  ax.legend(*zip(*unique))
legend_without_duplicate_labels(ax)
plt.savefig('s3dis-data-pipeline-output.png')