const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const DotenvWebpackPlugin = require('dotenv-webpack');

module.exports = {
  devtool: 'eval-source-map',
  entry: './src/index.ts',
  output: {
    filename: '[name][contenthash].js',
    path: path.resolve(__dirname, 'dist'),
    clean: true
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        include: [path.resolve(__dirname, 'src')],
        use: 'ts-loader',
      },
      {
        test: /\.scss$/,  // Process SCSS and output a separate CSS file
        use: [
          MiniCssExtractPlugin.loader, // Extracts CSS into separate file
          'css-loader', 
          'sass-loader'
        ],
        include: path.resolve(__dirname, 'src'),
      },
    ]
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: 'Object search in images',
      filename: 'index.html',
      template: 'src/template.html',
    }),
    new MiniCssExtractPlugin({
      filename: 'styles.css',
    }),
    new DotenvWebpackPlugin(),
  ],
  mode: 'development',
  devServer: {
    static: {
      directory: path.join(__dirname, 'dist'),
    },
    compress: true,
    port: 8000,
    hot: true,
    open: true,
  },
};