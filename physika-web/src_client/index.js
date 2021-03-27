import React from 'react';
import ReactDOM from 'react-dom';
import PhysikaWeb from './View'

window.localStorage.userID='localUser';

ReactDOM.render(
    <PhysikaWeb />,
    document.getElementById('root')
  );
