#coding:utf-8
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np
from IPython.display import display
import os


# 训练函数
def train_model(agent, env, episodes=1000):
    """
    训练模型
    """
    results = []
    for episode in range(episodes):
        state = env.reset()
        state = np.array(state)
        
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        
        if episode % 100 == 0:
            agent.update_target_network()
            
        results.append({
            'episode': episode,
            'reward': reward,
            'rt': info['rt'],
            'correct': reward > 0
        })
        
    return agent, results

def evaluate_model(agent, env, n_trials=100):
    results = []
    
    for _ in range(n_trials):
        state = env.reset()
        action = agent.act(state)
        _, reward, _, info = env.step(action)
        
        results.append({
            'congruent': env.word == env.ink_color,
            'correct': reward > 0,
            'rt': info['rt']
        })
    
    return results

def run_and_visualize(agent, env, episodes=1000, n_trials=200):
    """
    运行模型并可视化结果
    """
    # 训练模型
    print("开始训练模型...")
    agent, training_results = train_model(agent, env, episodes=episodes)
    
    # 评估模型
    print("评估模型表现...")
    eval_results = evaluate_model(agent, env, n_trials=n_trials)
    
    # 转换结果为DataFrame
    training_df = pd.DataFrame(training_results)
    eval_df = pd.DataFrame(eval_results)
    eval_df.to_csv('stroop_eval.csv', index=True, encoding='utf-8') # save
    
    # 创建图表
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 训练过程中的准确率变化
    plt.subplot(2, 2, 1)
    window_size = 50
    rolling_acc = training_df['correct'].rolling(window=window_size).mean()
    rolling_acc.to_csv('stroop_training.csv', index=True, encoding='utf-8') # save
    plt.plot(rolling_acc)
    plt.title(f'training acc (window size={window_size})')
    plt.xlabel('training epochs')
    plt.ylabel('acc')
    
    # 2. 一致性条件下的反应时间箱线图
    plt.subplot(2, 2, 2)
    seaborn.boxplot(data=eval_df, x='congruent', y='rt', hue='correct')
    plt.title('Reaction time (RT)')
    plt.xlabel('Congruency')
    plt.ylabel('RTs')
    
    # 3. 反应时间直方图
    plt.subplot(2, 2, 3)
    seaborn.histplot(data=eval_df, x='rt', hue='congruent', bins=20)
    plt.title('Reaction time (RT)')
    plt.xlabel('RTs')
    plt.ylabel('Frequency')
    
    # 4. 准确率条形图
    plt.subplot(2, 2, 4)
    acc_by_cond = eval_df.groupby('congruent')['correct'].mean()
    acc_by_cond.plot(kind='bar')
    plt.title('Accuracy')
    plt.xlabel('Congruency')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('./stroop_fig.eps')
    plt.show()
    
    # 打印统计数据
    print("\nModel performance statistics:")
    print("-" * 40)
    
    # 总体准确率
    print(f"总体准确率: {eval_df['correct'].mean():.3f}")
    
    # 按条件统计
    stats = eval_df.groupby('congruent').agg({
        'correct': ['mean', 'std'],
        'rt': ['mean', 'std']
    }).round(3)
    
    print("\nStatistics by condition:")
    print(stats)
    
    return agent, eval_df

def plot_learning_curve(training_results, window_size=50):
    """绘制学习曲线"""
    df = pd.DataFrame(training_results)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 准确率曲线
    rolling_acc = df['correct'].rolling(window=window_size).mean()
    ax1.plot(rolling_acc)
    ax1.set_title(f'Accuracy learning curve (window size={window_size})')
    ax1.set_xlabel('training epochs')
    ax1.set_ylabel('Accuracy')
    
    # 奖励曲线
    rolling_reward = df['reward'].rolling(window=window_size).mean()
    ax2.plot(rolling_reward)
    ax2.set_title(f'Reward learning curve (window size={window_size})')
    ax2.set_xlabel('training epochs')
    ax2.set_ylabel('Reward')
    
    plt.tight_layout()
    plt.savefig('./stroop_loss.png')
    plt.show()

def compare_with_human_data(model_results, human_data=None):
    """比较模型结果和人类数据"""
    if human_data is None:
        # 模拟一些人类数据用于演示
        human_data = pd.DataFrame({
            'congruent': np.random.choice([True, False], size=100),
            'correct': np.random.choice([True, False], size=100, p=[0.95, 0.05]),
            'rt': np.random.normal(0.6, 0.1, size=100)
        })
        # 调整不一致条件的RT
        human_data.loc[~human_data['congruent'], 'rt'] += 0.2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 比较准确率
    accuracy_data = pd.DataFrame({
        'Model': [model_results.groupby('congruent')['correct'].mean()[True],
                 model_results.groupby('congruent')['correct'].mean()[False]],
        'Human': [human_data.groupby('congruent')['correct'].mean()[True],
                 human_data.groupby('congruent')['correct'].mean()[False]]
    }, index=['Congruent', 'Incongruent'])
    
    accuracy_data.plot(kind='bar', ax=axes[0])
    axes[0].set_title('Accuracy comparison')
    axes[0].set_ylabel('Accuracy')
    
    # 比较反应时间
    rt_data = pd.DataFrame({
        'Model': [model_results.groupby('congruent')['rt'].mean()[True],
                 model_results.groupby('congruent')['rt'].mean()[False]],
        'Human': [human_data.groupby('congruent')['rt'].mean()[True],
                 human_data.groupby('congruent')['rt'].mean()[False]]
    }, index=['Congruent', 'Incongruent'])
    
    rt_data.plot(kind='bar', ax=axes[1])
    axes[1].set_title('Reaction time comparison')
    axes[1].set_ylabel('RTs')
    
    plt.tight_layout()
    plt.show()


# from evaluation_module import run_and_visualize
from paradigm_module.stroop_paradigm_module import StroopAgent, StroopEnvironment

if __name__ == "__main__":

    env = StroopEnvironment()
    agent = StroopAgent(state_size=8, action_size=4, base_dir='stroop_saved_models')

    # 运行模型并可视化结果
    agent, results = run_and_visualize(agent, env, episodes=1000, n_trials=200)
    
    # 如果有人类数据，可以进行比较
    # compare_with_human_data(results, human_data)

    # 保存模型
    model_name = "stroop_1000.zip"
    agent.save(model_name=model_name)