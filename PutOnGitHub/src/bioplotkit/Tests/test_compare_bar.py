from bioplotkit.plots import compare_bar
from bioplotkit.colors import PreColorDict

import unittest


class MyTestCase(unittest.TestCase):

    @staticmethod
    def generate_compare_bar_data():
        import string
        lower_alpha = list(string.ascii_lowercase)
        upper_alpha = list(string.ascii_uppercase)
        twenty_num = list(range(1, 21))

        d = {
            'DataA': lower_alpha[10:] + upper_alpha[10:] + twenty_num[10:],
            'DataB': lower_alpha,
            'DataC': upper_alpha,
            'DataD': twenty_num,
            'DataE': lower_alpha[10:] + upper_alpha[10:],
            'DataF': lower_alpha[10:] + twenty_num[10:],
            'DataG': lower_alpha[10:] + upper_alpha[10:] + twenty_num[10:],
            'DataH': lower_alpha[10:] + upper_alpha[10:] + twenty_num,
        }
        return d

    def test_compare_bar(self):
        d = self.generate_compare_bar_data()
        compare_bar.comp_bar(d, base_key='DataA', comp_keys=[f'Data{_}' for _ in ['B', 'C', 'D', 'E', 'F', 'G', 'H']],
                             base_x_pos=1,
                             bar_width=0.4,
                             share_color=PreColorDict['Blue'][75], new_color=PreColorDict['Red'][60], loss_color=PreColorDict['Grey'][60],
                             bar_edge_color='k', bar_edge_width=0.3,
                             anno_rotation=90, anno_fontsize=12, anno_ha='center', anno_va='center',
                             label_name=None, label_shift=0.1,
                             ylabel='Number', title='Compare 8 dataset',
                             ax=None, save=None)

        # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
