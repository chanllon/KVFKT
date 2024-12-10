import datetime
import os

class ModelConfigFactory():
    @staticmethod
    def create_model_config(args):
        if args.dataset == 'Junyi':
            return JunyiConfig(args).get_args()
        elif args.dataset == 'assist2017':
            return Assist2017Config(args).get_args()
        elif args.dataset == 'assist2012':
            return Assist2012Config(args).get_args()
        elif args.dataset == 'fsai':
            return FSAIConfig(args).get_args()
        elif args.dataset == 'NeurIPS':
            return NeurIPSConfig(args).get_args()
        elif args.dataset == 'EdNet':
            return EdNetConfig(args).get_args()
        elif args.dataset == 'Statics2011':
            return Statics2011Config(args).get_args()
        else:
            raise ValueError("The '{}' is not available".format(args.dataset))


#  相当于是一个接口
class ModelConfig():
    def __init__(self, args):
        self.default_setting = self.get_default_setting()    #初始化一个空的字典
        self.init_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M")

        self.args = args
        self.args_dict = vars(self.args)
        for arg in self.args_dict.keys():
            self._set_attribute_value(arg, self.args_dict[arg])     #key与value传入

        #初始化设置三个目录
        self.set_result_log_dir()
        self.set_checkpoint_dir()
        self.set_tensorboard_dir() 
    
    def get_args(self):
        return self.args

    def get_default_setting(self):
        default_setting = {}
        return default_setting

    #判断值是否空，如果空则拿默认的
    def _set_attribute_value(self, arg, arg_value):
        self.args_dict[arg] = arg_value \
            if arg_value is not None \
            else self.default_setting.get(arg)

    def _get_model_config_str(self):
        model_config = 'b' + str(self.args.batch_size) \
                    + '_m' + str(self.args.memory_size) \
                    + '_q' + str(self.args.key_memory_state_dim) \
                    + '_qa' + str(self.args.value_memory_state_dim) \
                    + '_f' + str(self.args.summary_vector_output_dim)
        return model_config

    def set_result_log_dir(self):
        result_log_dir = os.path.join(
            './results',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('result_log_dir', result_log_dir)

    def set_checkpoint_dir(self):
        checkpoint_dir = os.path.join(
            './models',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('checkpoint_dir', checkpoint_dir)
    
    def set_tensorboard_dir(self):
        tensorboard_dir = os.path.join(
            './tensorboard',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('tensorboard_dir', tensorboard_dir)
        

#继承ModelConfig
class JunyiConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            #开始都是50
            'n_epochs': 12,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.0003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 1332,
            'data_dir': './data/Junyi',
            'data_name': 'Junyi',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            'forget_cycle': 60000,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None,
            # parameter for the forget matrix
            'max_random_time': 1533080600.,
            'min_random_time': 1533080400.
        }
        return default_setting

'''
ok
'''
class Assist2017Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 20,
            'batch_size': 32,
            'train': True,
            'show': True,
            #'learning_rate': 0.003,
            'learning_rate': 0.0002,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 150,
            'data_dir': './data/ASSISTments2017',
            'data_name': 'ASSISTments2017',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 150,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            'forget_cycle': 600,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None,
            # parameter for the forget matrix
            'max_random_time' : 1095421300.,
            'min_random_time' : 1095420000.
        }
        return default_setting


# best
class Assist2012Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 10,
            'batch_size': 50,
            'train': True,
            'show': True,
            #'learning_rate': 0.003,
            'learning_rate': 0.0001,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 500,
            'data_dir': './data/ASSISTments2012',
            'data_name': 'ASSISTments2012',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            'forget_cycle': 60000,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None,
            # parameter for the forget matrix
            'max_random_time': 1346428800.,
            'min_random_time': 1346428000.
        }
        return default_setting


class FSAIConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 10,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 2267,
            'data_dir': './data/fsaif1tof3',
            'data_name': 'fsaif1tof3',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 150,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            'forget_cycle': 60000000000,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None,
            # parameter for the forget matrix
            'max_random_time': 1535485000.,
            'min_random_time': 1535484800.
        }
        return default_setting


class NeurIPSConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            #开始都是50
            'n_epochs': 20,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.0001,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 1100,
            'data_dir': './data/NeurIPS',
            'data_name': 'NeurIPS',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            'forget_cycle': 600000,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None,
            # parameter for the forget matrix
            'max_random_time': 1568000000.,
            'min_random_time': 1400000000.
        }
        return default_setting



class EdNetConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            #开始都是50
            'n_epochs': 20,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.0003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 1000,
            'data_dir': './data/NeurIPS',
            'data_name': 'NeurIPS',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            'forget_cycle': 6000000,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None,
            # parameter for the forget matrix
            'max_random_time': 1500957736.,
            'min_random_time': 1500957636.
        }
        return default_setting


class Statics2011Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            #开始都是50
            'n_epochs': 10,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 1223,
            'data_dir': './data/NeurIPS',
            'data_name': 'NeurIPS',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            'forget_cycle': 6000000,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None,
            # parameter for the forget matrix
            'max_random_time': 1313991370.,
            'min_random_time': 1313990000.
        }
        return default_setting