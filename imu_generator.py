import numpy as np
import pandas as pd

# -------------------------------
# CONFIGURATION
# -------------------------------
SAMPLING_RATE = 30  # frames per second
DT = 1.0 / SAMPLING_RATE

# -------------------------------
# LOAD SKELETON DATA
# -------------------------------
# Expected format: frame, joint, x, y, z
# Example columns: ['frame', 'joint', 'x', 'y', 'z']

def load_skeleton(csv_file):
    df = pd.read_csv(csv_file)
    return df


# -------------------------------
# COMPUTE VELOCITY & ACCELERATION
# -------------------------------
def compute_kinematics(df):
    imu_data = []

    joints = df['joint'].unique()

    for joint in joints:
        joint_data = df[df['joint'] == joint].sort_values('frame')

        positions = joint_data[['x', 'y', 'z']].values

        # Velocity
        velocity = np.gradient(positions, DT, axis=0)

        # Acceleration (IMU accelerometer)
        acceleration = np.gradient(velocity, DT, axis=0)

        # Gyroscope (angular velocity approximation)
        # Using cross product between consecutive velocity vectors
        angular_velocity = np.zeros_like(velocity)

        for i in range(1, len(velocity)):
            angular_velocity[i] = np.cross(velocity[i-1], velocity[i])

        # Store results
        for i in range(len(positions)):
            imu_data.append({
                'frame': joint_data.iloc[i]['frame'],
                'joint': joint,
                'acc_x': acceleration[i][0],
                'acc_y': acceleration[i][1],
                'acc_z': acceleration[i][2],
                'gyro_x': angular_velocity[i][0],
                'gyro_y': angular_velocity[i][1],
                'gyro_z': angular_velocity[i][2]
            })

    return pd.DataFrame(imu_data)


# -------------------------------
# ADD NOISE (REALISTIC IMU)
# -------------------------------
def add_noise(df, acc_noise=0.05, gyro_noise=0.02):
    noisy_df = df.copy()

    for col in ['acc_x', 'acc_y', 'acc_z']:
        noisy_df[col] += np.random.normal(0, acc_noise, len(df))

    for col in ['gyro_x', 'gyro_y', 'gyro_z']:
        noisy_df[col] += np.random.normal(0, gyro_noise, len(df))

    return noisy_df


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def generate_imu_dataset(input_csv, output_csv):
    print("Loading skeleton data...")
    df = load_skeleton(input_csv)

    print("Computing IMU signals...")
    imu_df = compute_kinematics(df)

    print("Adding sensor noise...")
    imu_df = add_noise(imu_df)

    print("Saving IMU dataset...")
    imu_df.to_csv(output_csv, index=False)

    print("IMU dataset generated successfully!")


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    input_file = "skeleton_data.csv"   # your keypoint file
    output_file = "imu_dataset.csv"

    generate_imu_dataset(input_file, output_file)