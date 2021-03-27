const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
/*
const plugins = [
    new HtmlWebpackPlugin({
        inject: 'body',
    }),
];
*/
module.exports = {
    plugins: [
        new HtmlWebpackPlugin({
            template: './static/view/index.html',
        }),
    ],
    entry: path.join(__dirname, './src_client/index.js'),
    output: {
        path: path.join(__dirname, './dist'),
        filename: 'physika-web.js',
        libraryTarget: 'umd',
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                include: path.join(__dirname, 'src_client'),
                exclude: /node_modules/,
                use: [
                    {
                        loader: 'babel-loader',
                        options: {
                            presets: ['@babel/preset-env', '@babel/preset-react'],
                            plugins: ["@babel/plugin-proposal-class-properties"],
                        },
                    },
                ],
            },
            // {
            //     test: /\.js$/,
            //     include: /node_modules(\/|\\)vtk.js(\/|\\)/,
            //     use: [
            //         {
            //             loader: 'babel-loader',
            //             options: {
            //                 presets: ['@babel/preset-env', '@babel/preset-react'],
            //             },
            //         },
            //     ],
            // },
            {
                test: /\.worker\.js$/,
                loader: 'worker-loader',
                options: {
                    inline: true,
                    fallback: false
                },
            },
            {
                test: /\.css$/,
                exclude: /\.module\.css$/,
                use: [
                    { loader: 'style-loader' },
                    { loader: 'css-loader' },
                ],
            },
            {
                test: /\.module\.css$/,
                use: [
                    { loader: 'style-loader' },
                    {
                        loader: 'css-loader',
                        options: {
                            modules: true,
                        },
                    },
                ],
            },
            {
                test: /\.glsl$/,
                loader: 'shader-loader',
            },
            {
                test: /\.svg$/,
                use: [{ loader: 'raw-loader' }],
            },
        ]
    },
    // node: {
    //     fs: "empty",
    //     net: 'empty',
    // },
};
