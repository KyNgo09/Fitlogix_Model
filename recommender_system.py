import pandas as pd
import numpy as np
import json
import io

class WorkoutRecommender:
    def __init__(self, exercise_csv_path):
        # --- 1. ĐỊNH NGHĨA MAP ---
        self.bodypart_map = {
            'pectorals': 'chest', 'quads': 'quadriceps', 'deltoids': 'shoulders', 'abs': 'abdominals',
            'abdominal': 'abdominals', 'bicep': 'biceps', 'tricep': 'triceps', 'gluteus': 'glutes',
            'hamstring': 'hamstrings', 'calf': 'calves', 'forearm': 'forearms', 'trap': 'traps',
            'latissimus dorsi': 'lats', 'lower back': 'lower back', 'middle back': 'middle back',
            'neck': 'neck', 'adductors': 'adductors', 'abductors': 'abductors'
        }
        
        self.difficulty_map = {'beginner': 1, 'intermediate': 2, 'expert': 3, 'advanced': 3}
        
        # === FIX LỖI: ĐỊNH NGHĨA DANH SÁCH DỤNG CỤ ===
        self.valid_equipment = [
            'Body Only', 'Barbell', 'Dumbbell', 'Cable', 'Machine', 
            'Kettlebell', 'Bands', 'E-Z Curl Bar', 'Medicine Ball', 'Exercise Ball', 'Foam Roll'
        ]
        
        self.major_muscles = ['chest', 'shoulders', 'lats', 'middle back', 'traps', 'quadriceps', 'glutes', 'abdominals', 'lower back']
        self.minor_muscles = ['triceps', 'biceps', 'forearms', 'calves', 'adductors', 'abductors', 'hamstrings']
        
        self.movement_map = {
            'push': ['chest', 'shoulders', 'triceps'],
            'pull': ['biceps', 'lats', 'middle back', 'traps', 'forearms'],
            'legs': ['quadriceps', 'hamstrings', 'glutes', 'calves', 'adductors', 'abductors'],
            'core': ['abdominals', 'lower back']
        }

        # --- 2. LOAD DỮ LIỆU ---
        try:
            self.raw_df = pd.read_csv(exercise_csv_path)
            # Rename cột
            self.raw_df = self.raw_df.rename(columns={
                'workout_id': 'ID', 'name': 'Title', 'category': 'BodyPart', 
                'level': 'Level', 'equipment': 'Equipment'
            })
        except Exception as e:
            print(f"⚠️ Error loading CSV: {e}. Using dummy data.")
            dummy_csv = """ID,Title,BodyPart,Equipment,Level,Rating
1,Push Up,Chest,Body Only,Beginner,9.0
2,Bench Press,Chest,Barbell,Intermediate,9.5
3,Dumbbell Press,Chest,Dumbbell,Intermediate,9.2
4,Machine Press,Chest,Machine,Beginner,8.5
5,Squat,Quadriceps,Barbell,Intermediate,9.5
6,Goblet Squat,Quadriceps,Dumbbell,Beginner,9.0
7,Bodyweight Squat,Quadriceps,Body Only,Beginner,8.0
"""
            self.raw_df = pd.read_csv(io.StringIO(dummy_csv))

        self.df = self._preprocess_data(self.raw_df)

    def _preprocess_data(self, df):
        required_cols = ['Title', 'BodyPart', 'Equipment', 'Level']
        if 'Rating' not in df.columns: df['Rating'] = 10.0
            
        df = df.dropna(subset=required_cols)
        df = df.drop_duplicates(subset=['Title', 'BodyPart'])
        df = df.copy()
        
        df['BodyPart_Clean'] = df['BodyPart'].str.lower().replace(self.bodypart_map)
        df['Level_Score'] = df['Level'].str.lower().map(self.difficulty_map).fillna(1)
        df['Equipment'] = df['Equipment'].astype(str).str.title()
        df.loc[df['Equipment'].isin(['None', 'Nan', '']), 'Equipment'] = 'Body Only'
        # Chuẩn hóa tên Body Only
        df.loc[df['Equipment'].str.lower() == 'body only', 'Equipment'] = 'Body Only'
        return df

    def _calculate_score(self, exercise_row, user_profile, target_muscles):
        # 1. Muscle Match
        muscle_match = 1.0 if exercise_row['BodyPart_Clean'] in target_muscles else 0.0
        
        # 2. Difficulty Fit
        diff = exercise_row['Level_Score'] - user_profile['fitness_level']
        if diff > 0: difficulty_fit = max(0, 1 - (diff * 0.5))
        else: difficulty_fit = max(0, 1 - (abs(diff) * 0.2))
        if user_profile['Age'] > 50 and diff > 0: difficulty_fit *= 0.5 

        # 3. Equipment Score (Ưu tiên dụng cụ có sẵn)
        equipment_score = 1.0
        # Nếu bài tập dùng dụng cụ (ko phải Body Only) và dụng cụ đó có trong list của user
        if exercise_row['Equipment'] != 'Body Only' and exercise_row['Equipment'] in user_profile['available_equipment']:
            equipment_score = 1.5 # Ưu tiên cao
        elif exercise_row['Equipment'] == 'Body Only':
            equipment_score = 1.0 # Bình thường
        else:
            equipment_score = 0.0 # Không có đồ -> Điểm thấp

        final_score = (0.5 * muscle_match) + (0.3 * difficulty_fit) + (0.2 * equipment_score)
        return final_score

    def _get_prescription(self, goal, fitness_level, gender, equipment):
        # Cấu hình Sets/Reps chuẩn khoa học
        presets = {
            "lose_fat": {
                1: {"sets": 3, "reps": 15, "weight": "40-50% 1 Rep Max"},
                2: {"sets": 4, "reps": 15, "weight": "50-60% 1 Rep Max"},
                3: {"sets": 5, "reps": 12, "weight": "60-70% 1 Rep Max"}, 
            },
            "gain_muscle": {
                1: {"sets": 3, "reps": 12, "weight": "60% 1 Rep Max"},
                2: {"sets": 4, "reps": 10, "weight": "65-75% 1 Rep Max"},
                3: {"sets": 5, "reps": 8, "weight": "75-85% 1 Rep Max"},
            },
            "maintain": {
                1: {"sets": 3, "reps": 12, "weight": "50-60% 1 Rep Max"},
                2: {"sets": 3, "reps": 12, "weight": "60-70% 1 Rep Max"},
                3: {"sets": 4, "reps": 10, "weight": "65-75% 1 Rep Max"},
            }
        }
        
        if goal == "strength": goal = "gain_muscle"
        goal_key = goal if goal in presets else "gain_muscle"
        
        # Ép kiểu level
        try: fitness_level = int(fitness_level)
        except: fitness_level = 1
            
        base_preset = presets[goal_key].get(fitness_level, presets[goal_key][1])
        result = base_preset.copy()
        
        if equipment == 'Body Only': 
            result['weight'] = 'Bodyweight'
            # Tăng reps cho bodyweight (trừ Advanced)
            if fitness_level < 3:
                result['reps'] = int(result['reps'] * 1.5)
        
        if gender == 'Female' and equipment != 'Body Only':
            result['sets'] = max(1, result['sets'] - 1)
            if "1RM" in str(result['weight']):
                try:
                    base_val = float(result['weight'].split('%')[0].split('-')[0])
                    new_val = max(0, base_val - 10)
                    result['weight'] = f"{new_val:.0f}% 1RM"
                except: pass
        return pd.Series(result)

    def _map_input_to_profile(self, api_data):
        level_map = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        raw_level = str(api_data.get('predicted_level', '')).strip().title()
        fitness_level = level_map.get(raw_level, 1)

        # --- LOGIC EQUIPMENT (Lọc cứng) ---
        workout_type = api_data.get('loai_hinh_tap_luyen', 'Gym')
        user_selected_tools = api_data.get('danh_sach_dung_cu', [])
        
        available_equipment = []

        if workout_type == 'Gym':
            # GYM: Lấy tất cả dụng cụ TRỪ Body Only
            # copy() để tránh sửa list gốc
            available_equipment = [e for e in self.valid_equipment if e != 'Body Only']
        else:
            # HOME: Mặc định Body Only
            available_equipment = ['Body Only']
            # Thêm dụng cụ nếu user chọn
            if user_selected_tools and len(user_selected_tools) > 0:
                extras = [e.title() for e in user_selected_tools]
                valid_extras = [e for e in extras if e in self.valid_equipment]
                available_equipment.extend(valid_extras)

        goal_vn = api_data.get('muc_tieu_chinh', 'Tăng cân')
        if goal_vn in ['Giảm cân', 'Giảm mỡ']: goal = 'lose_fat'
        elif goal_vn in ['Giữ dáng', 'Sức khỏe chung']: goal = 'maintain'
        else: goal = 'gain_muscle'

        gender_vn = api_data.get('gioi_tinh', 'Nam')
        gender = 'Male' if gender_vn == 'Nam' else 'Female'

        weight = api_data.get('can_nang_co_the', 70)
        height = api_data.get('chieu_cao', 170)
        height_m = height / 100
        bmi = weight / (height_m ** 2) if height_m > 0 else 0

        return {
            'fitness_level': fitness_level,
            'available_equipment': available_equipment,
            'goal': goal,
            'BMI': bmi,
            'Age': api_data.get('tuoi', 25),
            'Gender': gender
        }

    def recommend_from_api_json(self, api_data):
        user_profile = self._map_input_to_profile(api_data)
        return self._recommend_core(user_profile)

    def _recommend_core(self, user_profile):
        final_exercises = []
        
        # 1. Lọc Pool (Hard Filter)
        available_pool = self.df[
            self.df['Equipment'].isin(user_profile['available_equipment'])
        ].copy()

        # Fallback khẩn cấp: Nếu Gym mà không tìm thấy bài nào (hiếm), thì mới nhả Body Only
        if available_pool.empty and 'Body Only' not in user_profile['available_equipment']:
             available_pool = self.df[self.df['Equipment'] == 'Body Only'].copy()
        elif available_pool.empty:
             return pd.DataFrame()

        # Logic số lượng bài
        n_map = {
            'lose_fat': {1: 4, 2: 5, 3: 6},
            'gain_muscle': {1: 5, 2: 5, 3: 6},
            'maintain': {1: 4, 2: 5, 3: 5}
        }
        N = n_map.get(user_profile['goal'], n_map['maintain']).get(user_profile['fitness_level'], 4)
        
        major_count = round(N * 0.8) 
        if N >= 4: major_count = min(major_count, N - 1)
        
        for movement, muscles in self.movement_map.items():
            target_majors = [m for m in muscles if m in self.major_muscles]
            target_minors = [m for m in muscles if m in self.minor_muscles]

            current_pool = available_pool.copy()
            current_pool['score'] = current_pool.apply(
                lambda x: self._calculate_score(x, user_profile, muscles), axis=1
            )
            # Lọc bài có điểm > 0
            current_pool = current_pool[current_pool['score'] > 0]
            if current_pool.empty: continue

            selected_indices = []
            
            # Major (Phân bổ đều)
            pool_major = current_pool[current_pool['BodyPart_Clean'].isin(target_majors)]
            if not pool_major.empty and len(target_majors) > 0:
                quota_per_muscle = max(1, major_count // len(target_majors))
                for muscle in target_majors:
                    muscle_specific_pool = pool_major[pool_major['BodyPart_Clean'] == muscle]
                    if not muscle_specific_pool.empty:
                        top_candidates = muscle_specific_pool.sort_values('score', ascending=False).head(quota_per_muscle * 2)
                        count_to_pick = min(len(top_candidates), quota_per_muscle)
                        if len(selected_indices) < major_count:
                            count_to_pick = min(count_to_pick, major_count - len(selected_indices))
                            picked = top_candidates.sample(n=count_to_pick, random_state=None)
                            selected_indices.extend(picked.index.tolist())
                
                if len(selected_indices) < major_count:
                    remaining_needed = major_count - len(selected_indices)
                    remaining_pool = pool_major[~pool_major.index.isin(selected_indices)]
                    if not remaining_pool.empty:
                        top_rem = remaining_pool.sort_values('score', ascending=False).head(remaining_needed * 2)
                        picked_rem = top_rem.sample(n=min(len(top_rem), remaining_needed), random_state=None)
                        selected_indices.extend(picked_rem.index.tolist())

            # Minor
            pool_minor = current_pool[
                (current_pool['BodyPart_Clean'].isin(target_minors)) & 
                (~current_pool.index.isin(selected_indices))
            ]
            needed_total = N - len(selected_indices)
            if needed_total > 0 and not pool_minor.empty:
                top_minor = pool_minor.sort_values('score', ascending=False).head(needed_total * 2)
                selected_minor = top_minor.sample(n=min(len(top_minor), needed_total), random_state=None)
                selected_indices.extend(selected_minor.index.tolist())

            # Fallback
            if len(selected_indices) < N:
                needed = N - len(selected_indices)
                fallback_pool = current_pool[~current_pool.index.isin(selected_indices)]
                if not fallback_pool.empty:
                    top_fallback = fallback_pool.sort_values('score', ascending=False).head(needed * 2)
                    selected_fallback = top_fallback.sample(n=min(len(top_fallback), needed))
                    selected_indices.extend(selected_fallback.index.tolist())

            if not selected_indices: continue
            
            movement_result = current_pool.loc[selected_indices].copy()
            movement_result['movement'] = movement
            movement_result[['sets', 'reps', 'weight_recommendation']] = movement_result.apply(
                lambda x: self._get_prescription(
                    user_profile['goal'], user_profile['fitness_level'], 
                    user_profile['Gender'], x['Equipment']
                ), axis=1
            )
            final_exercises.append(movement_result)

        if not final_exercises: return pd.DataFrame()
        final_df = pd.concat(final_exercises).reset_index(drop=True)
        
        output_df = final_df[['ID', 'movement', 'Title', 'BodyPart', 'sets', 'reps', 'weight_recommendation', 'Level', 'Equipment']].rename(columns={
            'ID': 'workout_id', 'Title': 'exercise_name', 'BodyPart': 'primary_muscles', 
            'Level': 'difficulty', 'Equipment': 'equipment'
        })
        return output_df

