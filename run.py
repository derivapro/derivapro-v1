# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:55:04 2024

@author: minwuu01
"""

from derivapro import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)

