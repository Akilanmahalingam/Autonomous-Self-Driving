import numpy as np

class Car:
    def __init__(self, name, speed, distance, time):
        self.name = name
        self.speed = speed
        self.distance = distance
        self.time = time
        self.rank = -10
    
    def move(self):
        print('The car', self.name, 'is moving at', self.speed, 'km/h')
    
    def stop(self):
        print('The car', self.name, 'has stopped')
    
    def detail(self):
        print('The car', self.name, 'is moving at', self.speed, 'km/h')


class MarkovDecisionProcessAgent:
    def __init__(self, num_states, num_actions, transition_probabilities, rewards, discount_factor=0.9):
        self.discount_factor = discount_factor
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition_probabilities = transition_probabilities
        self.rewards = rewards
        self.values = np.zeros(num_states)
    
    def choose_action(self, state):
        # Ensure state is within the valid range
        state = min(max(0, state), self.num_states - 1)
        return np.argmax(self.values[state])
    
    # def update_values(self):
    #     new_values = np.zeros(self.num_states)
    #     for state in range(self.num_states):
    #         q_values = [sum(self.transition_probabilities[state, action, next_state] * (self.rewards[state, action, next_state] + self.discount_factor * self.values[next_state]) 
    #                        for next_state in range(self.num_states)) 
    #                     for action in range(self.num_actions)]
    #         new_values[state] = max(q_values)
    #     self.values = new_values

    def update_values(self):
        # Initialize a new array to store the updated values
        new_values = np.zeros(self.num_states)
        
        # Iterate over each state
        for state in range(self.num_states):
            # Calculate Q-values for all actions in the current state
            q_values = []
            for action in range(self.num_actions):
                action_q_value = 0
                
                # Calculate Q-value for each action using transition probabilities and rewards
                for next_state in range(self.num_states):
                    transition_probability = self.transition_probabilities[state, action, next_state]
                    immediate_reward = self.rewards[state, action, next_state]
                    discounted_future_value = self.discount_factor * self.values[next_state]
                    
                    action_q_value += transition_probability * (immediate_reward + discounted_future_value)
                
                q_values.append(action_q_value)
            
            # Update the value of the current state to the maximum Q-value
            new_values[state] = max(q_values)
        
        # Update the values array with the new values
        self.values = new_values



def simulate_environment(cars, agent, timeout, speed_rate):
    for i in range(len(cars) - 1):
        car_A = cars[i]
        car_B = cars[i + 1]
        car_B.speed = car_A.speed
        while (car_B.time - car_A.time <= timeout):
            # Ensure state is within the valid range
            state = min(max(0, int(car_B.distance)), agent.num_states - 1)
            action = agent.choose_action(state)
            if action == 0:  # Decelerate
                car_B.speed -= speed_rate
            car_B.time = car_B.distance / car_B.speed
            next_state = int(car_B.distance)
            agent.update_values()


car1 = Car('BMW', 10, 100, 10)
car2 = Car('Audi', 10, 100, 10)
car3 = Car('Benz', 10, 100, 10)
car4 = Car('Benz', 10, 100, 10)

cars = [car1, car2, car3, car4]

timeout = 2
speed_rate = 0.1

num_states = 100  # Number of states (simplified for the sake of example)
num_actions = 2  # 2 actions: 0 for decelerate, 1 for maintain speed

# Randomly initialize transition probabilities and rewards (for illustration purposes)
transition_probabilities = np.random.rand(num_states, num_actions, num_states)
transition_probabilities /= np.sum(transition_probabilities, axis=2, keepdims=True)
rewards = np.random.rand(num_states, num_actions, num_states)

agent = MarkovDecisionProcessAgent(num_states, num_actions, transition_probabilities, rewards)

simulate_environment(cars, agent, timeout, speed_rate)

print("Final speeds after collision avoidance:")
for car in cars:
    car.detail()
