'use strict';

var _ = require('underscore');

var _require = require('./request'),
    LCRequest = _require.request;

var _require2 = require('./utils'),
    getSessionToken = _require2.getSessionToken;

module.exports = function (AV) {
  /**
   * Contains functions to deal with Friendship in LeanCloud.
   * @class
   */
  AV.Friendship = {
    /**
     * Request friendship.
     * @since 4.8.0
     * @param {String | AV.User | Object} options if an AV.User or string is given, it will be used as the friend.
     * @param {AV.User | string} options.friend The friend (or friend's objectId) to follow.
     * @param {Object} [options.attributes] key-value attributes dictionary to be used as conditions of followeeQuery.
     * @param {*} [authOptions]
     * @return {Promise<void>}
     */
    request: function request(options, authOptions) {
      if (!AV.User.current()) {
        throw new Error('Please signin an user.');
      }
      var friend = void 0;
      var attributes = void 0;
      if (options.friend) {
        friend = options.friend;
        attributes = options.attributes;
      } else {
        friend = options;
      }
      var friendObject = _.isString(friend) ? AV.Object.createWithoutData('_User', friend) : friend;
      return LCRequest({
        method: 'POST',
        path: '/users/friendshipRequests',
        data: AV._encode({
          user: AV.User.current(),
          friend: friendObject,
          friendship: attributes
        }),
        authOptions: authOptions
      });
    },

    /**
     * Accept a friendship request.
     * @since 4.8.0
     * @param {AV.Object | string | Object} options if an AV.Object or string is given, it will be used as the request in _FriendshipRequest.
     * @param {AV.Object} options.request The request (or it's objectId) to be accepted.
     * @param {Object} [options.attributes] key-value attributes dictionary to be used as conditions of {@link AV#followeeQuery}.
     * @param {AuthOptions} [authOptions]
     * @return {Promise<void>}
     */
    acceptRequest: function acceptRequest(options) {
      var authOptions = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

      if (!getSessionToken(authOptions) && !AV.User.current()) {
        throw new Error('Please signin an user.');
      }
      var request = void 0;
      var attributes = void 0;
      if (options.request) {
        request = options.request;
        attributes = options.attributes;
      } else {
        request = options;
      }
      var requestId = _.isString(request) ? request : request.id;
      return LCRequest({
        method: 'PUT',
        path: '/users/friendshipRequests/' + requestId + '/accept',
        data: {
          friendship: AV._encode(attributes)
        },
        authOptions: authOptions
      });
    },

    /**
     * Decline a friendship request.
     * @param {AV.Object | string} request The request (or it's objectId) to be declined.
     * @param {AuthOptions} [authOptions]
     * @return {Promise<void>}
     */
    declineRequest: function declineRequest(request) {
      var authOptions = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : {};

      if (!getSessionToken(authOptions) && !AV.User.current()) {
        throw new Error('Please signin an user.');
      }
      var requestId = _.isString(request) ? request : request.id;
      return LCRequest({
        method: 'PUT',
        path: '/users/friendshipRequests/' + requestId + '/decline',
        authOptions: authOptions
      });
    }
  };
};