state_input = Input(shape=(input_shape))
cnn_feature = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu')(state_input)
cnn_feature = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')(cnn_feature)
cnn_feature = Convolution2D(64, 3, 3, activation='relu')(cnn_feature)
cnn_feature = Flatten()(cnn_feature)
cnn_feature = Dense(512, activation='relu')(cnn_feature)

distribution_list = []
for i in range(action_size): 
    distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))
    
num_samples = min(self.batch_size * self.timestep_per_train, len(self.memory))
replay_samples = random.sample(self.memory, num_samples)

state_inputs = np.zeros(((num_samples,) + self.state_size))
next_states = np.zeros(((num_samples,) + self.state_size))
m_prob = [np.zeros((num_samples, self.num_atoms)) for i in range(action_size)]
action, reward, done = [], [], []

for i in range(num_samples):
    state_inputs[i,:,:,:] = replay_samples[i][0]
    action.append(replay_samples[i][1])
    reward.append(replay_samples[i][2])
    next_states[i,:,:,:] = replay_samples[i][3]
    done.append(replay_samples[i][4])
    
z = self.model.predict(next_states)

optimal_action_idxs = []
z_concat = np.vstack(z)
q = np.sum(np.multiply(z_concat, np.array(self.z)), axis=1) # length (num_atoms x num_actions)
q = q.reshape((num_samples, action_size), order='F')
optimal_action_idxs = np.argmax(q, axis=1)

for i in range(num_samples):
    if done[i]: # Terminal State
        # Distribution collapses to a single point
        Tz = min(self.v_max, max(self.v_min, reward[i]))
        bj = (Tz - self.v_min) / self.delta_z
        m_l, m_u = math.floor(bj), math.ceil(bj)
        m_prob[action[i]][i][int(m_l)] += (m_u - bj)
        m_prob[action[i]][i][int(m_u)] += (bj - m_l)
    else:
        for j in range(self.num_atoms):
            Tz = min(self.v_max, max(self.v_min, reward[i] + self.gamma * self.z[j]))
            bj = (Tz - self.v_min) / self.delta_z
            m_l, m_u = math.floor(bj), math.ceil(bj)
            m_prob[action[i]][i][int(m_l)] += z_[optimal_action_idxs[i]][i][j] * (m_u - bj)
            m_prob[action[i]][i][int(m_u)] += z_[optimal_action_idxs[i]][i][j] * (bj - m_l)
            
loss = self.model.fit(state_inputs, m_prob, batch_size=self.batch_size, nb_epoch=1, verbose=0)
