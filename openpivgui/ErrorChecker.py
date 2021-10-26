import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
import openpiv.tools as piv_tls

# A lot of optimization could be done in this file.


# check number of images, image types, and window sizing
def check_PIVprocessing(self, roi):
    self.p = self
    '''Error checking'''
    # making sure there are 2 or more files loaded
    message = 'Please select two or more image files and/or apply an image frequence.'
    if len(self.p['fnames']) < 1:
        if self.p['warnings']:
            messagebox.showwarning(title='Error Message',
                                   message=message)
        raise Exception(message)

    # checking for images
    message = "Please supply image files in 'bmp', 'tif', 'jpg', 'jpeg', 'png', 'pgm'."
    test = self.p['files_a'][0]
    ext = test.split('.')[-1]
    if ext not in ['bmp', 'tif', 'TIF', 'tiff', 'jpg', 'jpeg', 'png', 'pgm']:
        if self.p['warnings']:
            messagebox.showwarning(title='Error Message',
                                   message=message)
        raise Exception(message)

    # checking interrogation window sizes in an inefficent manner (for now)
    test = piv_tls.imread(test)

    # making sure that the initial window is not too large
    xmin = roi[0]
    xmax = roi[1]
    ymin = roi[2]
    ymax = roi[3]
    try:
        if xmin and xmax and ymin and ymax != ('', ' '):
            test = test[int(ymin):int(ymax), int(xmin):int(xmax)]
    except: pass
    
    try:
        window_size = [int(self.p['corr_window_1']),
                       int(self.p['corr_window_1'])]
    except:
        window_size = [
            int(list(self.p['corr_window_1'].split(','))[0]),
            int(list(self.p['corr_window_1'].split(','))[1])
        ]

    try:
        overlap = [int(self.p['overlap_window_1']),
                   int(self.p['overlap_window_1'])]
    except:
        overlap = [
            int(list(self.p['overlap_window_1'].split(','))[0]),
            int(list(self.p['overlap_window_1'].split(','))[1])
        ]

    if ((test.shape[0] / window_size[0]) < 2.25 or
             (test.shape[1] / window_size[1]) < 2.25):
        message = ('Please lower your starting interrogation window size.')
        if self.p['warnings']:
            messagebox.showwarning(title='Error Message',
                                   message=message)
        raise ValueError(message)
    if overlap[0] > window_size[0] or overlap[1] > window_size[1]:
        message = 'Please make sure to make the overlap smaller than the window size(s).'
        if self.p['warnings']:
            messagebox.showwarning(title='Error Message',
                                   message=message)
        raise ValueError(message)
        
    # making sure each pass has a decreasing interrogation window
    old_window_size = window_size
    message = 'Please make sure that the windowing is decreasing with each pass.'
    message2 = 'Please make sure to make the overlap smaller than the window size(s).'
    for i in range(2, 7):
        if self.p['pass_%1d' % i]:
            try:
                window_size = [int(self.p[f'corr_window_{i}']),
                               int(self.p[f'corr_window_{i}'])]
            except:
                window_size = [
                    int(list(self.p[f'corr_window_{i}'].split(','))[0]),
                    int(list(self.p[f'corr_window_{i}'].split(','))[1])
                ]

            try:
                overlap = [int(self.p[f'overlap_window_{i}']),
                           int(self.p[f'overlap_window_{i}'])]
            except:
                overlap = [
                    int(list(self.p[f'overlap_window_{i}'].split(','))[0]),
                    int(list(self.p[f'overlap_window_{i}'].split(','))[1])
                ]
            
            if window_size[0] <= old_window_size[0] or window_size[1] <= old_window_size[1]:
                old_window_size = window_size
            else:
                if self.p['warnings']:
                    messagebox.showwarning(title='Error Message',
                                           message=message)
                raise ValueError(message)
                
            if overlap[0] > window_size[0] or overlap[1] > window_size[1]:
                if self.p['warnings']:
                    messagebox.showwarning(title='Error Message',
                                           message=message2)
                raise ValueError(message2)
        else:
            break

def check_processing(self):  # check for threads
    self = self
    message = 'Please stop all threads/processes to start processing.'
    checker = 0
    # check if any threads are alive
    try:
        if self.processing_thread.is_alive():
            if self.p['warnings']:
                messagebox.showwarning(title='Error Message',
                                       message=message)
            checker += 1
    except:
        pass

    try:
        if self.postprocessing_thread.is_alive():
            if self.p['warnings']:
                messagebox.showwarning(title='Error Message',
                                       message=message)
            checker += 1
    except:
        pass
    # if a thread is alive, an error shall be raised
    if checker != 0:
        # raising errors did not work in try statement for some reason
        raise Exception(message)
