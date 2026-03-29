import numpy as np


def tb_visualizer_pedes(tb_writer, lr, epoch, train_loss, valid_loss, train_result, valid_result,
                        train_gt, valid_gt, train_loss_mtr, valid_loss_mtr, model):
    tb_writer.add_scalars('train/lr', {'lr': lr}, epoch)
    tb_writer.add_scalars('train/losses', {'train': train_loss,
                                         'test': valid_loss}, epoch)

    tb_writer.add_scalars('train/perf', {'ma': train_result.ma,
                                         'pos_recall': np.mean(train_result.label_pos_recall),
                                         'neg_recall': np.mean(train_result.label_neg_recall),
                                         'Acc': train_result.instance_acc,
                                         'Prec': train_result.instance_prec,
                                         'Rec': train_result.instance_recall,
                                         'F1': train_result.instance_f1}, epoch)

    tb_writer.add_scalars('test/perf', {'ma': valid_result.ma,
                                        'pos_recall': np.mean(valid_result.label_pos_recall),
                                        'neg_recall': np.mean(valid_result.label_neg_recall),
                                        'Acc': valid_result.instance_acc,
                                        'Prec': valid_result.instance_prec,
                                        'Rec': valid_result.instance_recall,
                                        'F1': valid_result.instance_f1}, epoch)


