""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
import theano
import theano.tensor as T
import numpy as np
import gym
import h5py

# hyperparameters
H = 200 # number of hidden layer neurons
D = 80 * 80 # input dimensionality: 80x80 grid
batch_size = 10 # every how many episodes to do a param update?
save_batch_turn = 10 # how many batchs turns that to save weights
gamma = 0.99
random_threshold = 0.8 # how many percent that we dont use random behaber
learn_rate = 0.0001 # the nomal is 0.0001, and u can change it in training.
resume = False # resume from previous checkpoint?
render = True # if have the anime.

def build_model():
  model = Sequential()
  model.add(Dense(H, input_dim = D, activation = 'relu', init = 'lecun_uniform'))
  model.add(Dense(3, activation ='softmax'))
  model.load_weights('pong_weights.h5')
  optimizer = RMSprop(lr = learn_rate) #the normal lr is 0.0001
  model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)
  return model

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1 #this place maybe has some effect
  # the shape 80*80 full of 0.0 and 1.0 and ravel it to 1D 6400
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def action_outcome_dice(predict_arr):
  def get_action_onehot(i):
    if i == 0:
      return 2, [1.0, 0.0, 0.0] # get up
    elif i == 1:
      return 3, [0.0, 1.0, 0.0] # get down
    else:
      return 0, [0.0, 0.0, 1.0] # dont move

  def get_maxvalue_index():
    index = 0
    tmp_value = predict_arr[0]
    for i in xrange(len(predict_arr)):
      if predict_arr[i] > tmp_value:
        index = i
        tmp_value = predict_arr[i]
    return index
  
  if np.random.uniform() < random_threshold:
    return get_action_onehot(get_maxvalue_index())
  else:
    num = int(100*np.random.uniform())
    num = num - int(num/3)*3
    return get_action_onehot(num)

def start_training():
  #init the enveroments
  model = build_model()
  env = gym.make("Pong-v0")
  observation = env.reset()
  prev_x = None  # used in computing the difference frame
  xs, drs, ys = [], [], []
  data_buffer, label_buffer = [], []
  running_reward = None
  reward_sum = 0
  episode_number = 0

  while True:
    if render:
      env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)  # just for the first time
    prev_x = cur_x

    # is that Q-learn used in the model.predict ?
    action, y = action_outcome_dice(model.predict(np.array([x], np.float32))[0])

    xs.append(x)  # observation
    ys.append(y)
    observation, reward, reset, info = env.step(action) #comunicate whith everoments
    reward_sum += reward
    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if reset:  # an episode finished
      episode_number += 1

      #get labels from everomets
      discounted_epr = discount_rewards(drs)
      discounted_epr -= np.mean(discounted_epr)
      discounted_epr /= np.std(discounted_epr)  # the summer is 0. ?
      labels = (np.array(ys).T * discounted_epr).T

      #just make the memorys independent for training
      data_buffer += xs
      label_buffer += labels.tolist()

      # perform rmsprop parameter update every batch_size episodes
      if episode_number % batch_size == 0:
        model.fit(np.array(data_buffer, np.float32), np.array(label_buffer, np.float32), batch_size=512, nb_epoch=1)
        data_buffer, label_buffer = [], []
      if episode_number % (batch_size * save_batch_turn) == 0:
        model.save_weights('pong_weights.h5', overwrite=True)

      # boring book-keeping. That should be the ratio of the learning and the random.
      running_reward = reward_sum if running_reward is None else running_reward * gamma + reward_sum * (1 - gamma)
      print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)

      #resets
      reward_sum = 0
      prev_x = None
      observation = env.reset()  # reset env
      xs, drs, ys = [], [], []  # reset array memory

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
      print ('ep %d: game finished, reward: %f' % (episode_number, reward)) + ('' if reward == -1 else ' !!!!!!!!')

if __name__ == "__main__":
  start_training()